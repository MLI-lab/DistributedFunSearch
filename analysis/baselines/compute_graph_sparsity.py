#!/usr/bin/env python3
"""
Compute sparsity/density for all graphs in a directory.

Reads METIS file headers to get node/edge counts and computes density.
Very fast - only reads first line of each file.

Also includes KaMIS benchmark graphs from the paper for comparison:
    "Accelerating Local Search for the Maximum Independent Set Problem"
    (Dahlum et al., 2016) - arXiv:1602.01659

Usage:
    python compute_graph_sparsity.py --graph-dir /mnt/Graphs
    python compute_graph_sparsity.py --graph-dir /mnt/Graphs --output sparsity_results.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


# KaMIS benchmark graphs from Table 1 of the paper (arXiv:1602.01659)
# Format: (name, nodes, edges, category)
KAMIS_BENCHMARKS = [
    # Huge instances (from Table 1)
    ("it-2004", 41_291_594, 1_027_474_947, "KaMIS-huge"),
    ("sk-2005", 50_636_154, 1_810_063_330, "KaMIS-huge"),
    ("uk-2007", 105_896_555, 1_154_392_916, "KaMIS-huge"),
    # Social networks and Web graphs
    ("amazon-2008", 735_323, 3_523_472, "KaMIS-social"),
    ("as-Skitter-big", 1_696_415, 11_095_298, "KaMIS-social"),
    ("dewiki-2013", 1_532_354, 33_093_029, "KaMIS-social"),
    ("enwiki-2013", 4_206_785, 91_939_728, "KaMIS-social"),
    ("eu-2005", 862_664, 22_217_686, "KaMIS-social"),
    ("hollywood-2011", 2_180_759, 114_492_816, "KaMIS-social"),
    ("libimseti", 220_970, 17_233_144, "KaMIS-social"),
    ("ljournal-2008", 5_363_260, 49_514_271, "KaMIS-social"),
    ("orkut", 3_072_441, 117_185_082, "KaMIS-social"),
    ("web-Stanford", 281_903, 1_992_636, "KaMIS-social"),
    ("webbase-2001", 118_142_155, 854_809_761, "KaMIS-social"),
    ("wikilinks", 25_890_800, 543_159_884, "KaMIS-social"),
    ("youtube", 1_134_890, 2_987_624, "KaMIS-social"),
    # Road networks
    ("europe", 18_029_721, 22_217_686, "KaMIS-road"),
    ("USA-road", 23_947_347, 28_854_312, "KaMIS-road"),
    # Meshes
    ("buddha", 1_087_716, 1_631_574, "KaMIS-mesh"),
    ("bunny", 68_790, 103_017, "KaMIS-mesh"),
    ("dragon", 150_000, 225_000, "KaMIS-mesh"),
    ("feline", 41_262, 61_893, "KaMIS-mesh"),
    ("gameguy", 42_623, 63_850, "KaMIS-mesh"),
    ("venus", 5_672, 8_508, "KaMIS-mesh"),
]


def compute_density(nodes: int, edges: int) -> tuple:
    """Compute density and related stats."""
    max_edges = nodes * (nodes - 1) // 2
    density = edges / max_edges if max_edges > 0 else 0
    avg_degree = 2 * edges / nodes if nodes > 0 else 0
    sparsity = 1 - density
    return density, sparsity, avg_degree, max_edges


def get_kamis_benchmarks() -> list:
    """Get KaMIS benchmark graphs with computed density."""
    results = []
    for name, nodes, edges, category in KAMIS_BENCHMARKS:
        density, sparsity, avg_degree, max_edges = compute_density(nodes, edges)
        results.append({
            'name': name,
            'nodes': nodes,
            'edges': edges,
            'avg_degree': avg_degree,
            'density': density,
            'sparsity': sparsity,
            'category': category,
            'source': 'KaMIS paper'
        })
    return results


def get_graph_stats(metis_file: Path) -> dict:
    """Read METIS header and compute density."""
    with open(metis_file, 'r') as f:
        header = f.readline().strip()
        # Skip comments
        while header.startswith('%'):
            header = f.readline().strip()

        parts = header.split()
        nodes = int(parts[0])
        edges = int(parts[1])

    density, sparsity, avg_degree, max_edges = compute_density(nodes, edges)

    return {
        'nodes': nodes,
        'edges': edges,
        'avg_degree': avg_degree,
        'density': density,
        'sparsity': sparsity,
        'max_edges': max_edges
    }


def parse_filename(filename: str) -> dict:
    """Parse graph filename to extract parameters."""
    # Examples: graph_d_s1_n10_q2.metis, graph_ids_s1_n10_q2.metis
    name = filename.replace('.metis', '')
    parts = name.split('_')

    result = {}
    for part in parts:
        if part.startswith('s') and part[1:].isdigit():
            result['s'] = int(part[1:])
        elif part.startswith('n') and part[1:].isdigit():
            result['n'] = int(part[1:])
        elif part.startswith('q') and part[1:].isdigit():
            result['q'] = int(part[1:])

    if 'ids' in name:
        result['type'] = 'ids'
    elif '_d_' in name:
        result['type'] = 'deletion'
    else:
        result['type'] = 'unknown'

    return result


def main():
    parser = argparse.ArgumentParser(description='Compute graph sparsity')
    parser.add_argument('--graph-dir', type=str, default='/mnt/Graphs',
                        help='Root directory with graphs')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file (optional)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Output CSV file (optional)')
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)

    # Find all METIS files
    metis_files = list(graph_dir.glob('**/*.metis'))

    if not metis_files:
        print(f"No METIS files found in {graph_dir}")
        return

    print(f"Found {len(metis_files)} METIS files")
    print()

    # Collect results
    results = []

    for metis_file in sorted(metis_files):
        try:
            stats = get_graph_stats(metis_file)
            params = parse_filename(metis_file.name)

            results.append({
                'file': str(metis_file),
                'name': metis_file.name,
                **params,
                **stats
            })
        except Exception as e:
            print(f"Error reading {metis_file}: {e}")

    # Group by type and s
    by_type_s = defaultdict(list)
    for r in results:
        key = (r.get('type', 'unknown'), r.get('s', 0))
        by_type_s[key].append(r)

    # Print tables
    for (graph_type, s), graphs in sorted(by_type_s.items()):
        graphs = sorted(graphs, key=lambda x: x.get('n', 0))

        print("=" * 90)
        print(f"  {graph_type.upper()} GRAPHS, s={s}")
        print("=" * 90)
        print(f"{'n':>5} {'Nodes':>12} {'Edges':>15} {'Avg Degree':>12} {'Density':>10} {'Sparsity':>10}")
        print("-" * 90)

        densities = []
        for g in graphs:
            n = g.get('n', '?')
            print(f"{n:>5} {g['nodes']:>12,} {g['edges']:>15,} {g['avg_degree']:>12.1f} "
                  f"{g['density']*100:>9.2f}% {g['sparsity']*100:>9.2f}%")
            densities.append(g['density'])

        print("-" * 90)
        if densities:
            avg_density = sum(densities) / len(densities)
            min_density = min(densities)
            max_density = max(densities)
            print(f"{'Summary':>5} {'':>12} {'':>15} {'':>12} "
                  f"avg={avg_density*100:.1f}% range=[{min_density*100:.1f}%-{max_density*100:.1f}%]")
        print()

    # =========================================================================
    # COMBINED COMPARISON TABLE: Your graphs vs KaMIS benchmarks
    # =========================================================================
    print()
    print("=" * 100)
    print("  COMBINED COMPARISON: Your Graphs vs KaMIS Paper Benchmarks")
    print("  (Sorted by density, lowest to highest)")
    print("=" * 100)
    print()
    print("  Density formula: density = edges / (nodes × (nodes-1) / 2)")
    print("  KaMIS benchmarks from: 'Accelerating Local Search for the MIS Problem' (arXiv:1602.01659)")
    print()

    # Get KaMIS benchmarks
    kamis_results = get_kamis_benchmarks()

    # Prepare your graphs for comparison
    your_graphs = []
    for r in results:
        graph_type = r.get('type', 'unknown')
        s = r.get('s', '?')
        n = r.get('n', '?')
        name = f"{graph_type}_s{s}_n{n}"
        your_graphs.append({
            'name': name,
            'nodes': r['nodes'],
            'edges': r['edges'],
            'avg_degree': r['avg_degree'],
            'density': r['density'],
            'sparsity': r['sparsity'],
            'category': f"Your-{graph_type}-s{s}",
            'source': 'Your graphs'
        })

    # Combine and sort by density
    all_graphs = kamis_results + your_graphs
    all_graphs_sorted = sorted(all_graphs, key=lambda x: x['density'])

    # Print combined table
    print(f"{'Rank':>4} {'Name':<25} {'Nodes':>12} {'Edges':>15} {'Density':>12} {'Source':<20}")
    print("-" * 100)

    for i, g in enumerate(all_graphs_sorted, 1):
        density_str = f"{g['density']*100:.6f}%" if g['density'] < 0.01 else f"{g['density']*100:.2f}%"
        source_short = "KaMIS" if g['source'] == 'KaMIS paper' else "YOURS"
        print(f"{i:>4} {g['name']:<25} {g['nodes']:>12,} {g['edges']:>15,} {density_str:>12} {source_short:<6} {g['category']}")

    print("-" * 100)
    print()

    # Summary statistics
    kamis_densities = [g['density'] for g in kamis_results]
    your_densities = [g['density'] for g in your_graphs]

    print("  SUMMARY STATISTICS:")
    print(f"  {'':30} {'Min Density':>15} {'Max Density':>15} {'Avg Density':>15}")
    print(f"  {'-'*75}")
    print(f"  {'KaMIS paper benchmarks:':<30} {min(kamis_densities)*100:>14.6f}% {max(kamis_densities)*100:>14.4f}% {sum(kamis_densities)/len(kamis_densities)*100:>14.6f}%")
    print(f"  {'Your graphs:':<30} {min(your_densities)*100:>14.4f}% {max(your_densities)*100:>14.2f}% {sum(your_densities)/len(your_densities)*100:>14.4f}%")
    print()

    # Group your graphs by type-s and show ranges
    print("  YOUR GRAPHS BY TYPE:")
    by_category = defaultdict(list)
    for g in your_graphs:
        by_category[g['category']].append(g['density'])

    for cat in sorted(by_category.keys()):
        densities = by_category[cat]
        print(f"    {cat:<25} density range: {min(densities)*100:.4f}% - {max(densities)*100:.2f}%")
    print()

    # Recommendation
    print("  RECOMMENDATION FOR KAMIS:")
    print("    KaMIS is optimized for sparse graphs. Based on their benchmarks:")
    print(f"    - KaMIS benchmarks have density: {min(kamis_densities)*100:.6f}% - {max(kamis_densities)*100:.4f}%")
    print("    - Rule of thumb: KaMIS works well for density < ~1%")
    print()
    for cat in sorted(by_category.keys()):
        densities = by_category[cat]
        max_d = max(densities)
        if max_d < 0.01:
            status = "✓ Good fit for KaMIS"
        elif max_d < 0.10:
            status = "⚠ May work, try with timeout"
        else:
            status = "✗ Too dense for KaMIS"
        print(f"    {cat:<25} {status}")
    print()
    print("=" * 100)

    # Save JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    # Save CSV if requested
    if args.csv:
        with open(args.csv, 'w') as f:
            f.write("type,s,n,q,nodes,edges,avg_degree,density,sparsity\n")
            for r in sorted(results, key=lambda x: (x.get('type',''), x.get('s',0), x.get('n',0))):
                f.write(f"{r.get('type','')},{r.get('s','')},{r.get('n','')},{r.get('q','')},"
                        f"{r['nodes']},{r['edges']},{r['avg_degree']:.2f},"
                        f"{r['density']:.6f},{r['sparsity']:.6f}\n")
        print(f"CSV saved to {args.csv}")


if __name__ == '__main__':
    main()
