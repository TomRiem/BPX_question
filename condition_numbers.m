clear;

rng("default");

clear problem_data  

geo_name = 'geo_square.txt';

drchlt_sides = [1 2 3 4];

c_diff  = @(x, y) ones(size(x));

h = @(x, y, ind) x*0;

p = 3;
degree      = [p p];  
regularity  = [degree(1)-1 degree(2)-1];     
nsub_refine = [2 2];      
nquad       = [5 5];      
space_type  = 'standard'; 
truncated   = 1;           
nsub_coarse = [(2*degree(1) + 1) (2*degree(2) + 1)];


geometry = geo_load(geo_name);

[knots, zeta] = kntrefine (geometry.nurbs.knots, nsub_coarse-1, degree, regularity);
rule     = msh_gauss_nodes (nquad);
[qn, qw] = msh_set_quad_nodes (zeta, rule);
msh   = msh_cartesian (zeta, qn, qw, geometry);
space = sp_bspline (knots, degree, msh);


number_of_levels = 5;

cond = zeros(number_of_levels - 1, 1);
cond_precond = zeros(number_of_levels - 1, 1);

for it = 1:(number_of_levels - 1)
    hmsh     = hierarchical_mesh (msh, nsub_refine);
    hspace   = hierarchical_space (hmsh, space, space_type, truncated, regularity);
    
    % Local refinement
    for i_ref = 1:it
        marked = cell(i_ref,1);
        marked{i_ref} = []; 
    
        for j = 1:(degree(1) + 2^(i_ref-1))
            for i = 1:(degree(2) + 2^(i_ref-1))
                marked{i_ref} = [marked{i_ref}; sub2ind(hmsh.msh_lev{i_ref}.nel_dir, i, j)];
            end
        end
    
        [hmsh, new_cells] = hmsh_refine (hmsh, marked);
        marked_functions = compute_functions_to_deactivate (hmsh, hspace, marked, 'elements');
        hspace = hspace_refine (hspace, hmsh, marked_functions, new_cells);
    end
    
    % Assembling stiffness matrix
    stiff_mat = op_gradu_gradv_hier (hspace, hspace, hmsh, c_diff);
    [~, dirichlet_dofs] = sp_drchlt_l2_proj (hspace, hmsh, h, drchlt_sides);
    int_dofs = setdiff (1:hspace.ndof, dirichlet_dofs);
    stiff_mat_int = stiff_mat(int_dofs, int_dofs);

    % Computing condition number
    cond(it) =  max(real(eig(full(stiff_mat_int)))) / min(real(eig(full(stiff_mat_int))));

    % Assembling matrices for BPX preconditioner (Gauss-Seidel smoother)
    I = cell(hspace.nlevels, 1);
    D = cell(hspace.nlevels, 1);
    U = cell(hspace.nlevels, 1);
    L = cell(hspace.nlevels, 1);

    for i_lev = 1:hspace.nlevels
        indices = sort(find(any(hspace.Csub{i_lev} ~= 0, 1))); % Determining indices of functions in subspace 

        stiff_mat_subspace = stiff_mat(indices, indices); % Restriction of stiffness matrix to subspace

        D{i_lev} = diag(diag(stiff_mat_subspace)); % Diagonal entries 

        L{i_lev} = tril(stiff_mat_subspace); % Lower triangular part + diagonal entries

        U{i_lev} = triu(stiff_mat_subspace); % Upper triangular part + diagonal entries

        I{i_lev} = sparse(indices, 1:numel(indices), 1, hspace.ndof, numel(indices)); % Prolongation operator
        I{i_lev} = I{i_lev}(int_dofs, :); % restrcted to the reduced system 
    end

    % Appling BPX preconditioner to stiffness matrix 
    stiff_mat_precond = zeros(size(stiff_mat_int));

    for i_lev = 1:hspace.nlevels
        stiff_mat_tmp = I{i_lev}'*stiff_mat_int; % Mapping to subspace
        stiff_mat_tmp = L{i_lev} \ stiff_mat_tmp; % Solving with lower part
        stiff_mat_tmp = D{i_lev} * stiff_mat_tmp; % Multiplying with diagonal 
        stiff_mat_tmp = U{i_lev} \ stiff_mat_tmp; % Solving with upper part
        stiff_mat_precond = stiff_mat_precond + I{i_lev}*stiff_mat_tmp; % Mapping back to original space and summing up
    end

    % Computing condition number of preconditioned system
    cond_precond(it) =  max(real(eig(full(stiff_mat_precond)))) / min(real(eig(full(stiff_mat_precond))));
end


disp('Condition numbers of stiffness matrix (without preconditioner):')
disp(cond);
disp('Condition numbers of preconditioned stiffness matrix:')
disp(cond_precond);