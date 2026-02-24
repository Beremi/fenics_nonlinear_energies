/*
 * Custom greedy graph coloring: "most colored neighbors" vertex-selection
 * heuristic.  Faithful C translation of the MATLAB my_greedy_color2 function.
 *
 * The algorithm operates on A^2 (in CSC or CSR format — A^2 is symmetric so
 * both give identical structure).  It greedily colours vertices, always
 * choosing the uncoloured vertex among the current vertex's A^2-neighbours
 * that has the most already-coloured neighbours.  When all local neighbours
 * are already coloured, it falls back to a global search for the most
 * constrained uncoloured vertex.
 *
 * Compile:
 *   gcc -O3 -march=native -shared -fPIC -o custom_coloring.so custom_coloring.c
 */

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/*  custom_greedy_color                                                */
/*                                                                     */
/*  Input                                                              */
/*    n       – number of vertices                                     */
/*    indptr  – CSR/CSC pointers     (n+1 entries, int32)              */
/*    indices – CSR/CSC indices       (nnz entries, int32)              */
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
/*  Simple xoshiro128** PRNG (fast, no global state)                   */
/* ------------------------------------------------------------------ */
static inline unsigned int xoro_rotl(unsigned int x, int k)
{
    return (x << k) | (x >> (32 - k));
}

static unsigned int xoro_next(unsigned int s[4])
{
    unsigned int r = xoro_rotl(s[1] * 5, 7) * 9;
    unsigned int t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = xoro_rotl(s[3], 11);
    return r;
}

static void xoro_seed(unsigned int s[4], unsigned int seed)
{
    /* SplitMix32 to fill state from a single seed */
    for (int i = 0; i < 4; i++)
    {
        seed += 0x9e3779b9u;
        unsigned int z = seed;
        z = (z ^ (z >> 16)) * 0x85ebca6bu;
        z = (z ^ (z >> 13)) * 0xc2b2ae35u;
        z = z ^ (z >> 16);
        s[i] = z;
    }
}

/* ------------------------------------------------------------------ */
/*  custom_greedy_color_random                                         */
/*                                                                     */
/*  Same algorithm as custom_greedy_color but with randomised          */
/*  tie-breaking and random starting vertex.  Different seeds          */
/*  produce different colourings; running many in parallel and         */
/*  taking the best is a simple Monte-Carlo parallelisation.           */
/*                                                                     */
/*  Input                                                              */
/*    n, indptr, indices – A^2 in CSC (same as custom_greedy_color)    */
/*    seed               – PRNG seed (e.g. MPI rank)                   */
/*                                                                     */
/*  Output                                                             */
/*    colors  – 0-based colour for each vertex                         */
/*                                                                     */
/*  Returns: number of colours used                                    */
/* ------------------------------------------------------------------ */
int custom_greedy_color_random(int n,
                               const int *indptr,
                               const int *indices,
                               int *colors,
                               unsigned int seed)
{
    if (n == 0)
        return 0;
    if (n == 1)
    {
        colors[0] = 0;
        return 1;
    }

    /* PRNG state */
    unsigned int rng[4];
    xoro_seed(rng, seed);

    /* --- random start vertex --- */
    int start = (int)(xoro_next(rng) % (unsigned int)n);

    /* --- find max degree for colour-mask sizing --- */
    int max_deg = 0;
    for (int i = 0; i < n; i++)
    {
        int deg = indptr[i + 1] - indptr[i];
        if (deg > max_deg)
            max_deg = deg;
    }

    /* working memory */
    int mc = max_deg + 2;
    int *cn = (int *)calloc(n, sizeof(int));
    char *cmask = (char *)malloc(mc);
    char *done = (char *)calloc(n, sizeof(char));
    /* scratch for tie candidates (max possible = n) */
    int *ties = (int *)malloc(n * sizeof(int));
    if (!cn || !cmask || !done || !ties)
    {
        free(cn);
        free(cmask);
        free(done);
        free(ties);
        return -1;
    }

    for (int i = 0; i < n; i++)
        colors[i] = 1;

    cn[start] = 1;

    const int *nb = &indices[indptr[start]];
    int nnb = indptr[start + 1] - indptr[start];

    for (int step = 0; step < n; step++)
    {
        /* 1. Select: find max cn among UNCOLOURED local neighbours */
        int bv = -1;
        int ntie = 0;
        for (int j = 0; j < nnb; j++)
        {
            int v = nb[j];
            if (!done[v])
            {
                if (cn[v] > bv)
                {
                    bv = cn[v];
                    ntie = 0;
                    ties[ntie++] = v;
                }
                else if (cn[v] == bv)
                {
                    ties[ntie++] = v;
                }
            }
        }

        /* 2. Fallback: global search */
        if (ntie == 0)
        {
            bv = -1;
            for (int v = 0; v < n; v++)
            {
                if (!done[v])
                {
                    if (cn[v] > bv)
                    {
                        bv = cn[v];
                        ntie = 0;
                        ties[ntie++] = v;
                    }
                    else if (cn[v] == bv)
                    {
                        ties[ntie++] = v;
                    }
                }
            }
        }

        /* 3. Random tie-break */
        int bi = ties[xoro_next(rng) % (unsigned int)ntie];

        /* 4. Move to selected vertex */
        nb = &indices[indptr[bi]];
        nnb = indptr[bi + 1] - indptr[bi];

        /* 5. Build colour mask */
        memset(cmask, 1, mc);
        for (int j = 0; j < nnb; j++)
            cmask[colors[nb[j]]] = 0;

        /* 6. Increment coloured-neighbours for UNCOLOURED neighbours */
        for (int j = 0; j < nnb; j++)
        {
            if (!done[nb[j]])
                cn[nb[j]]++;
        }

        /* 7. Assign smallest available colour */
        int c = 1;
        while (!cmask[c])
            c++;
        colors[bi] = c;

        /* 8. Mark done */
        done[bi] = 1;
    }

    /* convert to 0-based */
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
    free(ties);
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

/* ------------------------------------------------------------------ */
/*  custom_greedy_color_omp                                            */
/*                                                                     */
/*  OpenMP-parallel version: block-partitions vertices among threads,   */
/*  each thread independently colours its local sub-graph of A^2, then */
/*  boundary conflicts are fixed greedily (sequential deterministic).   */
/*                                                                     */
/*  Advantage over MPI domain-decomposition: shared memory means A^2   */
/*  is computed only once (by Python) and shared across all threads.    */
/*                                                                     */
/*  Input                                                              */
/*    n                  - number of vertices                           */
/*    csc_indptr/indices - A^2 in CSC format (for greedy coloring)      */
/*    csr_indptr/indices - A^2 in CSR format (for boundary detection)   */
/*    nthreads           - OpenMP thread count (<=1 => serial fallback) */
/*                                                                     */
/*  Output                                                             */
/*    colors  - 0-based colour for each vertex                         */
/*                                                                     */
/*  Returns: number of colours used                                    */
/* ------------------------------------------------------------------ */
int custom_greedy_color_omp(int n,
                            const int *csc_indptr,
                            const int *csc_indices,
                            const int *csr_indptr,
                            const int *csr_indices,
                            int *colors,
                            int nthreads)
{
    if (n == 0)
        return 0;

#ifndef _OPENMP
    /* compiled without OpenMP - fall back to serial */
    (void)csr_indptr;
    (void)csr_indices;
    (void)nthreads;
    return custom_greedy_color(n, csc_indptr, csc_indices, colors);
#else
    if (nthreads <= 1)
        return custom_greedy_color(n, csc_indptr, csc_indices, colors);

    omp_set_num_threads(nthreads);

    /* --- block partition --- */
    int blk = n / nthreads;
    int rem_t = n % nthreads;
    int *starts = (int *)malloc((nthreads + 1) * sizeof(int));
    if (!starts)
        return custom_greedy_color(n, csc_indptr, csc_indices, colors);
    starts[0] = 0;
    for (int t = 0; t < nthreads; t++)
        starts[t + 1] = starts[t] + blk + (t < rem_t ? 1 : 0);

/* --- Phase 1: parallel local coloring --- */
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int s = starts[tid];
        int len = starts[tid + 1] - s;

        if (len > 0)
        {
            /* count local nnz (edges with both endpoints in this block) */
            int local_nnz = 0;
            for (int j = 0; j < len; j++)
            {
                int gj = s + j;
                for (int p = csc_indptr[gj]; p < csc_indptr[gj + 1]; p++)
                {
                    int r = csc_indices[p];
                    if (r >= s && r < s + len)
                        local_nnz++;
                }
            }

            /* build local CSC sub-graph (indices remapped to [0, len)) */
            int *lip = (int *)malloc((len + 1) * sizeof(int));
            int *lix = (int *)malloc((local_nnz > 0 ? local_nnz : 1) * sizeof(int));
            int *lc = (int *)calloc(len, sizeof(int));

            if (lip && lix && lc)
            {
                lip[0] = 0;
                int pos = 0;
                for (int j = 0; j < len; j++)
                {
                    int gj = s + j;
                    for (int p = csc_indptr[gj]; p < csc_indptr[gj + 1]; p++)
                    {
                        int r = csc_indices[p];
                        if (r >= s && r < s + len)
                            lix[pos++] = r - s;
                    }
                    lip[j + 1] = pos;
                }

                custom_greedy_color(len, lip, lix, lc);

                for (int j = 0; j < len; j++)
                    colors[s + j] = lc[j];
            }

            free(lip);
            free(lix);
            free(lc);
        }
    } /* end omp parallel */

    /* --- Phase 2: identify boundary vertices --- */
    /* A vertex is boundary if any A^2-neighbour is in a different block */
    int *boundary = (int *)malloc(n * sizeof(int));
    int n_boundary = 0;

    if (boundary)
    {
        for (int t = 0; t < nthreads; t++)
        {
            int s = starts[t];
            int e = starts[t + 1];
            for (int i = s; i < e; i++)
            {
                for (int p = csr_indptr[i]; p < csr_indptr[i + 1]; p++)
                {
                    int j = csr_indices[p];
                    if (j != i && (j < s || j >= e))
                    {
                        boundary[n_boundary++] = i;
                        break;
                    }
                }
            }
        }
    }

    /* --- Phase 3: fix boundary conflicts --- */
    int nc;
    if (boundary && n_boundary > 0)
    {
        nc = fix_coloring_conflicts(n, csr_indptr, csr_indices,
                                    n_boundary, boundary, colors);
    }
    else
    {
        nc = 0;
        for (int i = 0; i < n; i++)
            if (colors[i] > nc)
                nc = colors[i];
        nc++;
    }

    free(boundary);
    free(starts);
    return nc;
#endif /* _OPENMP */
}

/* ------------------------------------------------------------------ */
/*  Query whether the library was compiled with OpenMP support.         */
/*  Returns max threads if OpenMP is available, 0 otherwise.           */
/* ------------------------------------------------------------------ */
int custom_has_openmp(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 0;
#endif
}

/* ------------------------------------------------------------------ */
/*  build_A2_pattern                                                   */
/*                                                                     */
/*  Compute the sparsity pattern of A^2 directly (without numerical    */
/*  matrix multiplication).  For each vertex i, collect the union of   */
/*  all neighbours-of-neighbours (2-hop reachable set), including i.   */
/*                                                                     */
/*  Two-pass approach:                                                 */
/*    Pass 1: a2_indices == NULL — only compute row lengths (a2_indptr)*/
/*            and total nnz (*nnz_out).                                */
/*    Pass 2: a2_indices != NULL — fill the column indices.            */
/*                                                                     */
/*  Input:                                                             */
/*    n         – number of vertices                                   */
/*    ai, aj    – CSR indptr and indices of A  (int32)                 */
/*                                                                     */
/*  Output:                                                            */
/*    a2_indptr – CSR indptr of A^2  (n+1 entries)                     */
/*    a2_indices– CSR indices of A^2 (nnz entries, or NULL for pass 1) */
/*    nnz_out   – total number of nonzeros in A^2                      */
/*                                                                     */
/*  Returns 0 on success.                                              */
/* ------------------------------------------------------------------ */
int build_A2_pattern(int n,
                     const int *ai, const int *aj,
                     int *a2_indptr,
                     int *a2_indices,
                     int *nnz_out)
{
    /* marker array: marks which vertices are in current row's 2-hop set.
       marker[v] == i means v is in row i's set. Initialised to -1. */
    int *marker = (int *)malloc(n * sizeof(int));
    if (!marker) return -1;
    for (int v = 0; v < n; v++) marker[v] = -1;

    int nnz = 0;
    a2_indptr[0] = 0;

    for (int i = 0; i < n; i++)
    {
        int row_nnz = 0;

        /* Include self (diagonal of A^2) */
        marker[i] = i;
        if (a2_indices) a2_indices[nnz + row_nnz] = i;
        row_nnz++;

        /* For each neighbour j of i ... */
        for (int p = ai[i]; p < ai[i + 1]; p++)
        {
            int j = aj[p];
            /* j is a 1-hop neighbour: include it */
            if (marker[j] != i)
            {
                marker[j] = i;
                if (a2_indices) a2_indices[nnz + row_nnz] = j;
                row_nnz++;
            }
            /* For each neighbour k of j: k is a 2-hop neighbour */
            for (int q = ai[j]; q < ai[j + 1]; q++)
            {
                int k = aj[q];
                if (marker[k] != i)
                {
                    marker[k] = i;
                    if (a2_indices) a2_indices[nnz + row_nnz] = k;
                    row_nnz++;
                }
            }
        }

        /* Sort this row's entries for consistency */
        if (a2_indices)
        {
            /* Simple insertion sort (rows are typically small) */
            int *row = &a2_indices[nnz];
            for (int a = 1; a < row_nnz; a++)
            {
                int key = row[a];
                int b = a - 1;
                while (b >= 0 && row[b] > key)
                {
                    row[b + 1] = row[b];
                    b--;
                }
                row[b + 1] = key;
            }
        }

        nnz += row_nnz;
        a2_indptr[i + 1] = nnz;
    }

    *nnz_out = nnz;
    free(marker);
    return 0;
}
