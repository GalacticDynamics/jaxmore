window.BENCHMARK_DATA = {
  "lastUpdate": 1783888137863,
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
          "id": "1f7733d5f54f02c116b057ba0e22d60dfc9a61b4",
          "message": "build(deps): bump the actions group with 7 updates (#21)\n\n* build(deps): bump the actions group with 7 updates\n\nBumps the actions group with 7 updates:\n\n| Package | From | To |\n| --- | --- | --- |\n| [actions/checkout](https://github.com/actions/checkout) | `6` | `7` |\n| [hynek/build-and-inspect-python-package](https://github.com/hynek/build-and-inspect-python-package) | `2.17.0` | `2.18.0` |\n| [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish) | `1.13.0` | `1.14.0` |\n| [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) | `8.0.0` | `8.2.0` |\n| [codecov/codecov-action](https://github.com/codecov/codecov-action) | `6.0.0` | `7.0.0` |\n| [benchmark-action/github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark) | `1.22.0` | `1.22.1` |\n| [actions/github-script](https://github.com/actions/github-script) | `8` | `9` |\n\n\nUpdates `actions/checkout` from 6 to 7\n- [Release notes](https://github.com/actions/checkout/releases)\n- [Commits](https://github.com/actions/checkout/compare/v6...v7)\n\nUpdates `hynek/build-and-inspect-python-package` from 2.17.0 to 2.18.0\n- [Release notes](https://github.com/hynek/build-and-inspect-python-package/releases)\n- [Changelog](https://github.com/hynek/build-and-inspect-python-package/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/hynek/build-and-inspect-python-package/compare/fe0a0fb1925ca263d076ca4f2c13e93a6e92a33e...d44ca7d91762de7a7d5436ddae667c6da6d1c3df)\n\nUpdates `pypa/gh-action-pypi-publish` from 1.13.0 to 1.14.0\n- [Release notes](https://github.com/pypa/gh-action-pypi-publish/releases)\n- [Commits](https://github.com/pypa/gh-action-pypi-publish/compare/ed0c53931b1dc9bd32cbe73a98c7f6766f8a527e...cef221092ed1bacb1cc03d23a2d87d1d172e277b)\n\nUpdates `astral-sh/setup-uv` from 8.0.0 to 8.2.0\n- [Release notes](https://github.com/astral-sh/setup-uv/releases)\n- [Commits](https://github.com/astral-sh/setup-uv/compare/cec208311dfd045dd5311c1add060b2062131d57...fac544c07dec837d0ccb6301d7b5580bf5edae39)\n\nUpdates `codecov/codecov-action` from 6.0.0 to 7.0.0\n- [Release notes](https://github.com/codecov/codecov-action/releases)\n- [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/codecov/codecov-action/compare/57e3a136b779b570ffcdbf80b3bdc90e7fab3de2...fb8b3582c8e4def4969c97caa2f19720cb33a72f)\n\nUpdates `benchmark-action/github-action-benchmark` from 1.22.0 to 1.22.1\n- [Release notes](https://github.com/benchmark-action/github-action-benchmark/releases)\n- [Changelog](https://github.com/benchmark-action/github-action-benchmark/blob/master/CHANGELOG.md)\n- [Commits](https://github.com/benchmark-action/github-action-benchmark/compare/a60cea5bc7b49e15c1f58f411161f99e0df48372...52576c92bccf6ac60c8223ec7eb2565637cae9ba)\n\nUpdates `actions/github-script` from 8 to 9\n- [Release notes](https://github.com/actions/github-script/releases)\n- [Commits](https://github.com/actions/github-script/compare/v8...v9)\n\n---\nupdated-dependencies:\n- dependency-name: actions/checkout\n  dependency-version: '7'\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n  dependency-group: actions\n- dependency-name: hynek/build-and-inspect-python-package\n  dependency-version: 2.18.0\n  dependency-type: direct:production\n  update-type: version-update:semver-minor\n  dependency-group: actions\n- dependency-name: pypa/gh-action-pypi-publish\n  dependency-version: 1.14.0\n  dependency-type: direct:production\n  update-type: version-update:semver-minor\n  dependency-group: actions\n- dependency-name: astral-sh/setup-uv\n  dependency-version: 8.2.0\n  dependency-type: direct:production\n  update-type: version-update:semver-minor\n  dependency-group: actions\n- dependency-name: codecov/codecov-action\n  dependency-version: 7.0.0\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n  dependency-group: actions\n- dependency-name: benchmark-action/github-action-benchmark\n  dependency-version: 1.22.1\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n  dependency-group: actions\n- dependency-name: actions/github-script\n  dependency-version: '9'\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n  dependency-group: actions\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\n\n* docs: fix VAR_KEYWORD heading line break in README\n\nSigned-off-by: nstarman <nstarman@users.noreply.github.com>\n\n---------\n\nSigned-off-by: dependabot[bot] <support@github.com>\nSigned-off-by: nstarman <nstarman@users.noreply.github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>\nCo-authored-by: nstarman <nstarman@users.noreply.github.com>",
          "timestamp": "2026-07-12T16:26:29-04:00",
          "tree_id": "f6220545d6cb727d8b5504b6879ee44122e39095",
          "url": "https://github.com/GalacticDynamics/jaxmore/commit/1f7733d5f54f02c116b057ba0e22d60dfc9a61b4"
        },
        "date": 1783888137324,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop",
            "value": 26.861007154398585,
            "unit": "iter/sec",
            "range": "stddev: 0.0006942648707180631",
            "extra": "mean: 37.22868596296273 msec\nrounds: 27"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop_jit",
            "value": 111997.25885741967,
            "unit": "iter/sec",
            "range": "stddev: 0.000001696649184750635",
            "extra": "mean: 8.928789956128032 usec\nrounds: 30546"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_single_positional",
            "value": 125162.76556345663,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010711432515964504",
            "extra": "mean: 7.9895965505253015 usec\nrounds: 54443"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_two_positionals",
            "value": 121190.90834514075,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010025756346699949",
            "extra": "mean: 8.251444053477101 usec\nrounds: 62524"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_with_kwonly",
            "value": 120849.34305786011,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011864611079968456",
            "extra": "mean: 8.27476570990726 usec\nrounds: 56127"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_outs_only",
            "value": 250498.77899605068,
            "unit": "iter/sec",
            "range": "stddev: 8.23217089588755e-7",
            "extra": "mean: 3.992035426311462 usec\nrounds: 80985"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_bind_free_pos_only",
            "value": 123672.08863306507,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013861254077627554",
            "extra": "mean: 8.085898856022387 usec\nrounds: 61714"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_varargs_bind_free",
            "value": 122972.15681547597,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010003633487610998",
            "extra": "mean: 8.131922102501097 usec\nrounds: 61260"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_pos_only_default_omitted",
            "value": 88507.52484159428,
            "unit": "iter/sec",
            "range": "stddev: 0.000001484988322670854",
            "extra": "mean: 11.298474358985217 usec\nrounds: 45084"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_static_path",
            "value": 2656.135876144347,
            "unit": "iter/sec",
            "range": "stddev: 0.000035986645355070115",
            "extra": "mean: 376.48676371617034 usec\nrounds: 1367"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_kw_path",
            "value": 2152.4171034223555,
            "unit": "iter/sec",
            "range": "stddev: 0.000029276391369940994",
            "extra": "mean: 464.59396666658813 usec\nrounds: 1110"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_general_path",
            "value": 2369.2967512046735,
            "unit": "iter/sec",
            "range": "stddev: 0.000017446599955091834",
            "extra": "mean: 422.0661677316478 usec\nrounds: 1252"
          }
        ]
      }
    ]
  }
}