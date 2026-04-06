window.BENCHMARK_DATA = {
  "lastUpdate": 1775498093151,
  "repoUrl": "https://github.com/GalacticDynamics/jaxmore",
  "entries": {
    "jaxmore Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "nstarman@users.noreply.github.com",
            "name": "Nathaniel Starkman",
            "username": "nstarman"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5f2db852f7961b0ecc474761b56a43cb3105790c",
          "message": "🧱 infra: set up unit, usage, and benchmark tests (#18)\n\nSigned-off-by: nstarman <nstarman@users.noreply.github.com>",
          "timestamp": "2026-04-06T13:51:59-04:00",
          "tree_id": "5542ec5ba7e2ad6b4db25a419558146915ac9d7b",
          "url": "https://github.com/GalacticDynamics/jaxmore/commit/5f2db852f7961b0ecc474761b56a43cb3105790c"
        },
        "date": 1775498092878,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop",
            "value": 28.421862325560138,
            "unit": "iter/sec",
            "range": "stddev: 0.0015677496672958792",
            "extra": "mean: 35.18418281481462 msec\nrounds: 27"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop_jit",
            "value": 103098.64306415794,
            "unit": "iter/sec",
            "range": "stddev: 0.000002020995515334754",
            "extra": "mean: 9.699448705427708 usec\nrounds: 28658"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_static_path",
            "value": 2261.5436695870476,
            "unit": "iter/sec",
            "range": "stddev: 0.00003111084653058876",
            "extra": "mean: 442.1758524709796 usec\nrounds: 1376"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_kw_path",
            "value": 1760.8287851101222,
            "unit": "iter/sec",
            "range": "stddev: 0.000020995758596601017",
            "extra": "mean: 567.9143869387959 usec\nrounds: 1225"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_general_path",
            "value": 1877.899960369682,
            "unit": "iter/sec",
            "range": "stddev: 0.000014494730917783398",
            "extra": "mean: 532.5097295401938 usec\nrounds: 1283"
          }
        ]
      }
    ]
  }
}