using Ferrite, Tensors
using BlockArrays, SparseArrays, LinearAlgebra

function create_cook_grid(nx, ny)
  corners = [
    Vec{2}((0.0, 0.0)),
    Vec{2}((48.0, 44.0)),
    Vec{2}((48.0, 60.0)),
    Vec{2}((0.0, 44.0)),
  ]
  grid = generate_grid(Triangle, (nx, ny), corners)
  # Facets for boundary conditions
  addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
  addfacetset!(grid, "traction", x -> norm(x[1]) ≈ 48.0)
  return grid
end

function create_values(interpolation_u, interpolation_p)
  # Quadrature rules
  qr = QuadratureRule{RefTriangle}(3)
  facet_qr = FacetQuadratureRule{RefTriangle}(3)

  # Cell and FacetValues for u
  cellvalues_u = CellValues(qr, interpolation_u)
  facetvalues_u = FacetValues(facet_qr, interpolation_u)

  # Cellvalues for p
  cellvalues_p = CellValues(qr, interpolation_p)

  return cellvalues_u, cellvalues_p, facetvalues_u
end


function create_dofhandler(grid, ipu, ipp)
  dh = DofHandler(grid)
  add!(dh, :u, ipu) # Displacement
  add!(dh, :p, ipp) # Pressure
  close!(dh)
  return dh
end

function create_bc(dh)
  dbc = ConstraintHandler(dh)
  add!(dbc, Dirichlet(:u, getfacetset(dh.grid, "clamped"), x -> zero(x), [1, 2]))
  close!(dbc)
  return dbc
end

struct LinearElasticity{T}
  G::T
  K::T
end


function doassemble(
  cellvalues_u::CellValues,
  cellvalues_p::CellValues,
  facetvalues_u::FacetValues,
  K::SparseMatrixCSC, grid::Grid, dh::DofHandler, mp::LinearElasticity
)
  f = zeros(ndofs(dh))
  assembler = start_assemble(K, f)
  nu = getnbasefunctions(cellvalues_u)
  np = getnbasefunctions(cellvalues_p)

  fe = BlockedArray(zeros(nu + np), [nu, np]) # local force vector
  ke = BlockedArray(zeros(nu + np, nu + np), [nu, np], [nu, np]) # local stiffness matrix

  # Traction vector
  t = Vec{2}((0.0, 1 / 16))
  # Cache ϵdev outside the element routine to avoid some unnecessary allocations
  ϵdev = [zero(SymmetricTensor{2,2}) for _ in 1:getnbasefunctions(cellvalues_u)]

  for cell in CellIterator(dh)
    fill!(ke, 0)
    fill!(fe, 0)
    assemble_up!(ke, fe, cell, cellvalues_u, cellvalues_p, facetvalues_u, grid, mp, ϵdev, t)
    assemble!(assembler, celldofs(cell), ke, fe)
  end

  return K, f
end

function assemble_up!(Ke, fe, cell, cellvalues_u, cellvalues_p, facetvalues_u, grid, mp, ϵdev, t)
  nu = getnbasefunctions(cellvalues_u)
  np = getnbasefunctions(cellvalues_p)
  u□, p□ = 1, 2
  reinit!(cellvalues_u, cell)
  reinit!(cellvalues_p, cell)

  # We only assemble the lower half of the stiffness matrix and the symemtrize it
  for q_point in 1:getnquadpoints(cellvalues_u)
    for i in 1:nu
      ϵdev[i] = dev(symmetric(shape_gradient(cellvalues_u, q_point, i)))
    end
    dΩ = getdetJdV(cellvalues_u, q_point)
    for i in 1:nu
      # TODO: The following two lines from the tutorial are obsolete
      divδu = shape_divergence(cellvalues_u, q_point, i)
      δu = shape_value(cellvalues_u, q_point, i)
      for j in 1:i
        Ke[BlockIndex((u□, u□), (i, j))] += 2 * mp.G * ϵdev[i] ⊡ ϵdev[j] * dΩ
      end
    end

    for i in 1:np
      δp = shape_value(cellvalues_p, q_point, i)
      for j in 1:np
        divδu = shape_divergence(cellvalues_u, q_point, j)
        Ke[BlockIndex((p□, u□), (i, j))] += -δp * divδu * dΩ
      end
      for j in 1:i
        p = shape_value(cellvalues_p, q_point, j)
        Ke[BlockIndex((p□, p□), (i, j))] += -1 / mp.K * δp * p * dΩ
      end
    end
  end

  symmetrize_lower!(Ke)

  # We integrate the Neumann boundary values using the FacetValues.
  # We loop over all the facets in the cell, then check if the
  # is in our 'traction' facetset.
  for facet in 1:nfacets(cell)
    if (cellid(cell), facet) ∈ getfacetset(grid, "traction")
      reinit!(facetvalues_u, cell, facet)
      for q_point in 1:getnquadpoints(facetvalues_u)
        dΓ = getdetJdV(facetvalues_u, q_point)
        for i in 1:nu
          δu = shape_value(facetvalues_u, q_point, i)
          fe[i] += (δu ⋅ t) * dΓ
        end
      end
    end
  end
end

function symmetrize_lower!(Ke)
  for i in 1:size(Ke, 1)
    for j in (i+1):size(Ke, 1)
      Ke[i, j] = Ke[j, i]
    end
  end
  return
end;

function compute_stresses(
  cellvalues_u::CellValues, cellvalues_p::CellValues,
  dh::DofHandler, mp::LinearElasticity, a::Vector
)
  ae = zeros(ndofs_per_cell(dh)) # Local solution vector
  u_range = dof_range(dh, :u) # Local range of dofs corresponding to u
  p_range = dof_range(dh, :p) # Local range of dofs corresponding to p
  # Allocate storage for the stresses
  σ = zeros(SymmetricTensor{2,3}, getncells(dh.grid))
  # Loop over the cells and compute the cell-average stress
  for cell in CellIterator(dh)
    # Update cellvalues
    reinit!(cellvalues_u, cell)
    reinit!(cellvalues_p, cell)
    # Extract the cell local part of the solution
    for (i, I) in pairs(celldofs(cell))
      ae[i] = a[I]
    end
    # Loop over the quadrature points
    σΩi = zero(SymmetricTensor{2,3}) # Stress integrated over the cell
    Ωi = 0.0 # Cell volume (Area)
    for qp in 1:getnquadpoints(cellvalues_u)
      dΩ = getdetJdV(cellvalues_u, qp)
      # Evaluate the strain and the pressure
      ϵ = function_symmetric_gradient(cellvalues_u, qp, ae, u_range)
      p = function_value(cellvalues_p, qp, ae, p_range)
      # Expand strain to 3D
      ϵ3D = SymmetricTensor{2,3}((i, j) -> i < 3 && j < 3 ? ϵ[i, j] : 0.0)
      # Compute the stress in this quadrature point
      σqp = 2 * mp.G * dev(ϵ3D) - one(ϵ3D) * p
      σΩi += σqp * dΩ
      Ωi += dΩ
    end
    # Store the value
    σ[cellid(cell)] = σΩi / Ωi
  end
  return σ
end


function solve(ν, interpolation_u, interpolation_p)
  # Material
  Emod = 1.0
  Gmod = Emod / 2(1 + ν)
  Kmod = Emod * ν / ((1 + ν) * (1 - 2ν))
  mp = LinearElasticity(Gmod, Kmod)

  # Grid, DofHandler, Boudary Condition
  n = 50
  grid = create_cook_grid(n, n)
  dh = create_dofhandler(grid, interpolation_u, interpolation_p)
  dbc = create_bc(dh)

  # Cellvalues
  cellvalues_u, cellvalues_p, facetvalues_u = create_values(interpolation_u, interpolation_p)

  # Assembly and solve
  K = allocate_matrix(dh)
  K, f = doassemble(cellvalues_u, cellvalues_p, facetvalues_u, K, grid, dh, mp)
  apply!(K, f, dbc)
  u = K \ f

  # Compute stress
  σ = compute_stresses(cellvalues_u, cellvalues_p, dh, mp, u)
  σvM = map(x -> √(3 / 2 * dev(x) ⊡ dev(x)), σ) # von Mises effective stress

  # Export the solution and the stress
  filename = "cook_" *
             (interpolation_u == Lagrange{RefTriangle,1}()^2 ? "linear" : "quadratic") *
             "_linear"

  VTKGridFile(filename, grid) do vtk
    write_solution(vtk, dh, u)
    for i in 1:3, j in 1:3
      σij = [x[i, j] for x in σ]
      write_cell_data(vtk, σij, "sigma_$(i)$(j)")
    end
    write_cell_data(vtk, σvM, "sigma von Mises")
  end
  return u
end

linear_p = Lagrange{RefTriangle,1}()
linear_u = Lagrange{RefTriangle,1}()^2
quadratic_u = Lagrange{RefTriangle,2}()^2

u1 = solve(0.5, linear_u, linear_p); # This should return garbage, due to LBB rule
u2 = solve(0.5, quadratic_u, linear_p); # This should work fine
