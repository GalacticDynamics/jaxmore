window.BENCHMARK_DATA = {
  "lastUpdate": 1778535156531,
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
      },
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
          "id": "30acacfcbf6239d7a1ae3cf87e3f701b3e91c459",
          "message": "✨ feat: structured decorator (#13)\n\nSigned-off-by: nstarman <nstarman@users.noreply.github.com>",
          "timestamp": "2026-04-06T15:47:45-04:00",
          "tree_id": "120356403888c1f06c2871246a10c3950d703511",
          "url": "https://github.com/GalacticDynamics/jaxmore/commit/30acacfcbf6239d7a1ae3cf87e3f701b3e91c459"
        },
        "date": 1775505026010,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop",
            "value": 28.661238774912093,
            "unit": "iter/sec",
            "range": "stddev: 0.0006539598453694073",
            "extra": "mean: 34.89032724137957 msec\nrounds: 29"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop_jit",
            "value": 99081.30638928752,
            "unit": "iter/sec",
            "range": "stddev: 0.000002356215941072417",
            "extra": "mean: 10.092721184670594 usec\nrounds: 28700"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_single_positional",
            "value": 119207.03437250502,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013409593678520966",
            "extra": "mean: 8.388766697065394 usec\nrounds: 50488"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_two_positionals",
            "value": 117143.59763887092,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013743093060539543",
            "extra": "mean: 8.536531403814228 usec\nrounds: 56076"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_with_kwonly",
            "value": 112081.933488852,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013209818185795076",
            "extra": "mean: 8.9220445157601 usec\nrounds: 55149"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_outs_only",
            "value": 234496.2586050915,
            "unit": "iter/sec",
            "range": "stddev: 8.499716040520067e-7",
            "extra": "mean: 4.264460362602508 usec\nrounds: 105065"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_bind_free_pos_only",
            "value": 110693.38657641734,
            "unit": "iter/sec",
            "range": "stddev: 0.000002689212429036965",
            "extra": "mean: 9.033963373319041 usec\nrounds: 60202"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_varargs_bind_free",
            "value": 117728.04965366232,
            "unit": "iter/sec",
            "range": "stddev: 0.000001577879393932597",
            "extra": "mean: 8.494152438113472 usec\nrounds: 59165"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_pos_only_default_omitted",
            "value": 80718.05412804072,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014609160040975745",
            "extra": "mean: 12.388802118714716 usec\nrounds: 44461"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_static_path",
            "value": 2099.39407277759,
            "unit": "iter/sec",
            "range": "stddev: 0.000032456510017905495",
            "extra": "mean: 476.32791430955905 usec\nrounds: 1202"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_kw_path",
            "value": 1827.1857938108894,
            "unit": "iter/sec",
            "range": "stddev: 0.000025612066717144997",
            "extra": "mean: 547.289719188512 usec\nrounds: 1282"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_general_path",
            "value": 1937.5879600045514,
            "unit": "iter/sec",
            "range": "stddev: 0.000030493914231479554",
            "extra": "mean: 516.1056017284764 usec\nrounds: 1273"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b035f4ddf48016f973fce13083846fa160214d43",
          "message": "build(deps): bump urllib3 from 2.6.3 to 2.7.0 (#19)\n\nBumps [urllib3](https://github.com/urllib3/urllib3) from 2.6.3 to 2.7.0.\n- [Release notes](https://github.com/urllib3/urllib3/releases)\n- [Changelog](https://github.com/urllib3/urllib3/blob/main/CHANGES.rst)\n- [Commits](https://github.com/urllib3/urllib3/compare/2.6.3...2.7.0)\n\n---\nupdated-dependencies:\n- dependency-name: urllib3\n  dependency-version: 2.7.0\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-05-11T17:29:40-04:00",
          "tree_id": "998bd4c858b553856b25325ec4338b21f2514c7d",
          "url": "https://github.com/GalacticDynamics/jaxmore/commit/b035f4ddf48016f973fce13083846fa160214d43"
        },
        "date": 1778535155652,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop",
            "value": 28.068052902890916,
            "unit": "iter/sec",
            "range": "stddev: 0.0004947738294519408",
            "extra": "mean: 35.62769399999967 msec\nrounds: 28"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop_jit",
            "value": 101859.02615625408,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022822158520719765",
            "extra": "mean: 9.817490287664612 usec\nrounds: 28469"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_single_positional",
            "value": 117732.20095283611,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016803230185398095",
            "extra": "mean: 8.493852929842049 usec\nrounds: 48929"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_two_positionals",
            "value": 116215.45650846469,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013649709690092482",
            "extra": "mean: 8.604707411936758 usec\nrounds: 39113"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_with_kwonly",
            "value": 112826.08713880085,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014430657809415971",
            "extra": "mean: 8.86319844425501 usec\nrounds: 53093"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_outs_only",
            "value": 229203.6208811714,
            "unit": "iter/sec",
            "range": "stddev: 9.473231065578967e-7",
            "extra": "mean: 4.362932819976876 usec\nrounds: 104406"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_bind_free_pos_only",
            "value": 120933.22811672158,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012849177558122128",
            "extra": "mean: 8.269025937477052 usec\nrounds: 58371"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_varargs_bind_free",
            "value": 117443.95354283234,
            "unit": "iter/sec",
            "range": "stddev: 0.000001314147075980662",
            "extra": "mean: 8.514699734076096 usec\nrounds: 60533"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_pos_only_default_omitted",
            "value": 84911.95137592456,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015421065627564308",
            "extra": "mean: 11.776905179964269 usec\nrounds: 44421"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_static_path",
            "value": 2244.5094219999773,
            "unit": "iter/sec",
            "range": "stddev: 0.000018428742586613183",
            "extra": "mean: 445.53165613755664 usec\nrounds: 1393"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_kw_path",
            "value": 1820.3664931864482,
            "unit": "iter/sec",
            "range": "stddev: 0.0000320220914478894",
            "extra": "mean: 549.3399289335175 usec\nrounds: 1182"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_general_path",
            "value": 1996.1422441446439,
            "unit": "iter/sec",
            "range": "stddev: 0.00002332972992297951",
            "extra": "mean: 500.9663028440664 usec\nrounds: 1301"
          }
        ]
      }
    ]
  }
}