/*
 * Custom greedy graph coloring: "most colored neighbors" vertex-selection
 * heuristic.  Faithful C translation of the MATLAB my_greedy_color2 function.
 *
 * The algorithm operates on A^2 (in CSC format, including diagonal) and
 * greedily colours vertices, always choosing the uncoloured vertex among the
 * current vertex's A^2-neighbours that has the most already-coloured
 * neighbours.  When all local neighbours are already coloured, it falls back
 * to a global search for the most constrained uncoloured vertex.
 *
 * Compile:
 *   gcc -O3 -march=native -shared -fPIC -o custom_coloring.so custom_coloring.c
 */

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  custom_greedy_color                                                */
/*                                                                     */
/*  Input                                                              */
/*    n       – number of vertices                                     */
/*    indptr  – CSC column pointers  (n+1 entries, int32)              */
/*    indices – CSC row indices      (nnz entries, int32)              */
/*                                                                     */
/*  Output                                                             */
/*    colors  – 0-based colour for each vertex (n entries, int32)      */
/*                                                                     */
/*  Returns: number of colours used                                    */
/* ------------------------------------------------------------------ */
int custom_greedy_color(int n,
                        const int *indptr,
                        const int *indices,
                        int *colors)
{
    if (n == 0)
        return 0;
    if (n == 1)
    {
        colors[0] = 0;
        return 1;
    }

    /* --- find start vertex (maximum degree in A^2) --- */
    int max_deg = 0, start = 0;
    for (int i = 0; i < n; i++)
    {
        int deg = indptr[i + 1] - indptr[i];
        if (deg > max_deg)
        {
            max_deg = deg;
            start = i;
        }
    }

    /* working memory */
    int mc = max_deg + 2;                         /* safe upper bound for colour mask  */
    int *cn = (int *)calloc(n, sizeof(int));      /* coloured-neighbours */
    char *cmask = (char *)malloc(mc);             /* colour availability */
    char *done = (char *)calloc(n, sizeof(char)); /* 1 = already coloured */
    if (!cn || !cmask || !done)
    {
        free(cn);
        free(cmask);
        free(done);
        return -1;
    }

    /* initialise: all colours = 1 (uncoloured sentinel, MATLAB convention) */
    for (int i = 0; i < n; i++)
        colors[i] = 1;

    /* prioritise start vertex so it is selected first */
    cn[start] = 1;

    /* current neighbour list (initially: neighbours of start vertex) */
    const int *nb = &indices[indptr[start]];
    int nnb = indptr[start + 1] - indptr[start];

    /* --- main loop: colour one vertex per iteration --- */
    for (int step = 0; step < n; step++)
    {

        /* 1. Select next vertex: max cn among current UNCOLOURED neighbours */
        int bv = -1, bi = -1;
        for (int j = 0; j < nnb; j++)
        {
            int v = nb[j];
            if (!done[v] && cn[v] > bv)
            {
                bv = cn[v];
                bi = v;
            }
        }

        /* 2. Fallback: global search if no uncoloured local neighbour found */
        if (bi == -1)
        {
            bv = -1;
            for (int v = 0; v < n; v++)
            {
                if (!done[v] && cn[v] > bv)
                {
                    bv = cn[v];
                    bi = v;
                }
            }
        }

        /* 3. Move to selected vertex */
        nb = &indices[indptr[bi]];
        nnb = indptr[bi + 1] - indptr[bi];

        /* 4. Build colour mask from neighbour colours */
        memset(cmask, 1, mc);
        for (int j = 0; j < nnb; j++)
            cmask[colors[nb[j]]] = 0;

        /* 5. Increment coloured-neighbours count for all UNCOLOURED neighbours */
        for (int j = 0; j < nnb; j++)
        {
            if (!done[nb[j]])
                cn[nb[j]]++;
        }

        /* 6. Assign smallest available colour (1-based during algorithm) */
        int c = 1;
        while (!cmask[c])
            c++;
        colors[bi] = c;

        /* 7. Mark vertex as coloured */
        done[bi] = 1;
    }

    /* convert to 0-based and compute number of colours */
    int nc = 0;
    for (int i = 0; i < n; i++)
    {
        colors[i]--;
        if (colors[i] >= nc)
            nc = colors[i] + 1;
    }

    free(cn);
    free(cmask);
    free(done);
    return nc;
}

/* ------------------------------------------------------------------ */
/*  fix_coloring_conflicts                                             */
/*                                                                     */
/*  After domain-decomposition parallel colouring, some boundary       */
/*  vertices may share a colour with an A^2-neighbour in a different   */
/*  partition.  This routine greedily re-colours those vertices until   */
/*  no conflicts remain.                                               */
/*                                                                     */
/*  Input                                                              */
/*    n                 – total number of vertices                      */
/*    row_ptr, col_idx  – CSR representation of full A^2               */
/*    n_boundary        – number of boundary vertices to check         */
/*    boundary          – sorted list of boundary vertex indices        */
/*                                                                     */
/*  In/Out                                                             */
/*    colors            – 0-based colour array (modified in-place)      */
/*                                                                     */
/*  Returns: final number of colours                                   */
/* ------------------------------------------------------------------ */
int fix_coloring_conflicts(int n,
                           const int *row_ptr,
                           const int *col_idx,
                           int n_boundary,
                           const int *boundary,
                           int *colors)
{
    if (n_boundary == 0)
        goto done;

    /* find max degree for scratch buffer sizing */
    int max_deg = 0;
    for (int k = 0; k < n_boundary; k++)
    {
        int i = boundary[k];
        int deg = row_ptr[i + 1] - row_ptr[i];
        if (deg > max_deg)
            max_deg = deg;
    }

    {
        char *used = (char *)calloc(max_deg + n_boundary + 2, sizeof(char));
        int *marked = (int *)malloc((max_deg + 1) * sizeof(int));
        if (!used || !marked)
        {
            free(used);
            free(marked);
            goto done;
        }

        int changed = 1, passes = 0;
        while (changed && passes < 50)
        {
            changed = 0;
            passes++;

            for (int k = 0; k < n_boundary; k++)
            {
                int i = boundary[k];
                int my_c = colors[i];
                int p0 = row_ptr[i];
                int p1 = row_ptr[i + 1];

                /* check for conflict */
                int conflict = 0;
                for (int p = p0; p < p1; p++)
                {
                    int j = col_idx[p];
                    if (j != i && colors[j] == my_c)
                    {
                        conflict = 1;
                        break;
                    }
                }

                if (conflict)
                {
                    /* mark used colours among neighbours */
                    int nm = 0;
                    for (int p = p0; p < p1; p++)
                    {
                        int j = col_idx[p];
                        if (j != i)
                        {
                            int c = colors[j];
                            if (!used[c])
                            {
                                used[c] = 1;
                                marked[nm++] = c;
                            }
                        }
                    }

                    /* find smallest unused colour */
                    int nc = 0;
                    while (used[nc])
                        nc++;
                    colors[i] = nc;

                    /* clear marks */
                    for (int m = 0; m < nm; m++)
                        used[marked[m]] = 0;

                    changed = 1;
                }
            }
        }

        free(used);
        free(marked);
    }

done:;
    int max_c = 0;
    for (int i = 0; i < n; i++)
    {
        if (colors[i] > max_c)
            max_c = colors[i];
    }
    return max_c + 1;
}
