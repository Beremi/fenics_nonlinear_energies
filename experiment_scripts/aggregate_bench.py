import json
import statistics

configs = ['np1', 'np4', 'np8', 'np16']
for cfg in configs:
    runs_by_level = {}
    for r in range(1, 4):
        with open(f'results/jax_petsc_bench/{cfg}_run{r}.json') as f:
            data = json.load(f)
        for res in data['results']:
            lvl = res['mesh_level']
            if lvl not in runs_by_level:
                runs_by_level[lvl] = {'setup': [], 'solve': [], 'iters': [], 'energy': [], 'dofs': [], 'n_colors': []}
            runs_by_level[lvl]['setup'].append(res['setup_time'])
            runs_by_level[lvl]['solve'].append(res['time'])
            runs_by_level[lvl]['iters'].append(res['iters'])
            runs_by_level[lvl]['energy'].append(res['energy'])
            runs_by_level[lvl]['dofs'].append(res['dofs'])
            runs_by_level[lvl]['n_colors'].append(res['n_colors'])

    print(f'\n=== {cfg} ===')
    for lvl in sorted(runs_by_level):
        d = runs_by_level[lvl]
        print(
            f'lvl={lvl} dofs={
                d["dofs"][0]} setup={
                statistics.median(
                    d["setup"]):.3f} solve={
                    statistics.median(
                        d["solve"]):.3f} iters={
                            d["iters"][0]} n_colors={
                                d["n_colors"][0]} J={
                                    d["energy"][0]:.4f}')
