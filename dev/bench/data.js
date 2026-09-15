window.BENCHMARK_DATA = {
  "lastUpdate": 1789505819317,
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
          "id": "cb486cc437ee0552faa621d188bf2b413661d728",
          "message": "build(deps): bump tornado from 6.5.5 to 6.5.7 (#23)\n\nBumps [tornado](https://github.com/tornadoweb/tornado) from 6.5.5 to 6.5.7.\n- [Changelog](https://github.com/tornadoweb/tornado/blob/master/docs/releases.rst)\n- [Commits](https://github.com/tornadoweb/tornado/compare/v6.5.5...v6.5.7)\n\n---\nupdated-dependencies:\n- dependency-name: tornado\n  dependency-version: 6.5.7\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-12T20:06:39-04:00",
          "tree_id": "c0dab84117c822588d7d9d131c11b745c368d09f",
          "url": "https://github.com/GalacticDynamics/jaxmore/commit/cb486cc437ee0552faa621d188bf2b413661d728"
        },
        "date": 1783901431049,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop",
            "value": 26.11291040830678,
            "unit": "iter/sec",
            "range": "stddev: 0.0011679282289314798",
            "extra": "mean: 38.29523344444555 msec\nrounds: 27"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop_jit",
            "value": 112036.42892398988,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015550521295243715",
            "extra": "mean: 8.92566828132697 usec\nrounds: 29727"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_closure",
            "value": 836.4776514371487,
            "unit": "iter/sec",
            "range": "stddev: 0.00015226576822517418",
            "extra": "mean: 1.1954892019911163 msec\nrounds: 703"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_carry",
            "value": 851.6515827123853,
            "unit": "iter/sec",
            "range": "stddev: 0.00005584506729445588",
            "extra": "mean: 1.1741890936375021 msec\nrounds: 833"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_via_step_kw",
            "value": 837.457985490635,
            "unit": "iter/sec",
            "range": "stddev: 0.0002913437306191186",
            "extra": "mean: 1.1940897541434725 msec\nrounds: 724"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_empty_batch_skipping",
            "value": 1374.8003411537168,
            "unit": "iter/sec",
            "range": "stddev: 0.000026749226543182553",
            "extra": "mean: 727.3783472884589 usec\nrounds: 1051"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_single_positional",
            "value": 123042.00866215353,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011330423804001741",
            "extra": "mean: 8.12730555095034 usec\nrounds: 54387"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_two_positionals",
            "value": 119697.0536378065,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011645231362216237",
            "extra": "mean: 8.354424520973744 usec\nrounds: 60699"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_with_kwonly",
            "value": 117530.55524392583,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024447362314028217",
            "extra": "mean: 8.50842572745934 usec\nrounds: 56158"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_outs_only",
            "value": 248053.14744083086,
            "unit": "iter/sec",
            "range": "stddev: 9.807729496087952e-7",
            "extra": "mean: 4.031394119836896 usec\nrounds: 104113"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_bind_free_pos_only",
            "value": 125817.83894083402,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011054905198518022",
            "extra": "mean: 7.94799853834917 usec\nrounds: 58838"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_varargs_bind_free",
            "value": 122587.100021019,
            "unit": "iter/sec",
            "range": "stddev: 0.000001947376333976446",
            "extra": "mean: 8.157465180500544 usec\nrounds: 61905"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_pos_only_default_omitted",
            "value": 88076.28701416461,
            "unit": "iter/sec",
            "range": "stddev: 0.000001313918911045885",
            "extra": "mean: 11.353793783781756 usec\nrounds: 45719"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_static_path",
            "value": 2633.073020670039,
            "unit": "iter/sec",
            "range": "stddev: 0.00002844729454649383",
            "extra": "mean: 379.7843782340414 usec\nrounds: 1121"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_kw_path",
            "value": 2193.459688132972,
            "unit": "iter/sec",
            "range": "stddev: 0.000020319146446455443",
            "extra": "mean: 455.90078787870476 usec\nrounds: 1155"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_general_path",
            "value": 2291.480278156162,
            "unit": "iter/sec",
            "range": "stddev: 0.00002887834829763913",
            "extra": "mean: 436.39913008749494 usec\nrounds: 1253"
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
          "id": "db2eb1461bb7e60a5fe509d4c616c6410a562c84",
          "message": "💚 ci(prek): replace pre-commit with prek (#22)",
          "timestamp": "2026-07-12T21:11:32-04:00",
          "tree_id": "f4be41e06dafd6b4220ba5600ada17b1bd427abc",
          "url": "https://github.com/GalacticDynamics/jaxmore/commit/db2eb1461bb7e60a5fe509d4c616c6410a562c84"
        },
        "date": 1783905276578,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop",
            "value": 28.699834429962078,
            "unit": "iter/sec",
            "range": "stddev: 0.000561628002755652",
            "extra": "mean: 34.843406586207315 msec\nrounds: 29"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop_jit",
            "value": 99504.07075408274,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025468447593625743",
            "extra": "mean: 10.049840096205001 usec\nrounds: 25359"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_closure",
            "value": 779.7115301351994,
            "unit": "iter/sec",
            "range": "stddev: 0.00012206425767072243",
            "extra": "mean: 1.2825256025476541 msec\nrounds: 785"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_carry",
            "value": 761.8063212708747,
            "unit": "iter/sec",
            "range": "stddev: 0.00010639146728629392",
            "extra": "mean: 1.312669601286271 msec\nrounds: 622"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_via_step_kw",
            "value": 767.9163822819207,
            "unit": "iter/sec",
            "range": "stddev: 0.00014461769543791238",
            "extra": "mean: 1.3022251160060234 msec\nrounds: 681"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_empty_batch_skipping",
            "value": 1177.0244331375034,
            "unit": "iter/sec",
            "range": "stddev: 0.00004038246651960478",
            "extra": "mean: 849.6000353487796 usec\nrounds: 1075"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_single_positional",
            "value": 114199.97875128775,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017383555689895396",
            "extra": "mean: 8.756569054867043 usec\nrounds: 46883"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_two_positionals",
            "value": 109932.37454913968,
            "unit": "iter/sec",
            "range": "stddev: 0.000017137413164515254",
            "extra": "mean: 9.096501409173152 usec\nrounds: 53577"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_with_kwonly",
            "value": 104614.56573148714,
            "unit": "iter/sec",
            "range": "stddev: 0.000012593896714843345",
            "extra": "mean: 9.558898352325883 usec\nrounds: 52013"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_outs_only",
            "value": 232021.89634527673,
            "unit": "iter/sec",
            "range": "stddev: 8.500285509950368e-7",
            "extra": "mean: 4.309938052190896 usec\nrounds: 98922"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_bind_free_pos_only",
            "value": 119180.75275070951,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013171908345291881",
            "extra": "mean: 8.390616579605776 usec\nrounds: 56648"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_varargs_bind_free",
            "value": 115933.17899392165,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013627262392723934",
            "extra": "mean: 8.625658406662252 usec\nrounds: 57627"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_pos_only_default_omitted",
            "value": 82033.91757581301,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018070717331229664",
            "extra": "mean: 12.190079781034878 usec\nrounds: 42930"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_static_path",
            "value": 2180.7501932133614,
            "unit": "iter/sec",
            "range": "stddev: 0.00002628121370258412",
            "extra": "mean: 458.5577949790242 usec\nrounds: 1434"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_kw_path",
            "value": 1668.9798604113603,
            "unit": "iter/sec",
            "range": "stddev: 0.000021442083588359493",
            "extra": "mean: 599.1684044369031 usec\nrounds: 1172"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_general_path",
            "value": 1875.6921589642623,
            "unit": "iter/sec",
            "range": "stddev: 0.000029285439721572344",
            "extra": "mean: 533.1365252132789 usec\nrounds: 1289"
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
          "id": "524d2455ff761ded9e18f88cd252d4bbf4b17f6c",
          "message": "💚 ci: bump hynek/build-and-inspect-python-package to v3.0.1 (#25)\n\nv2.x pins Twine 6, which rejects packaging metadata 2.5 emitted by\ncurrent build backends, failing the distribution build. v3 ships Twine 7.\n\nCo-authored-by: Claude Fable 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-09-07T11:49:32-04:00",
          "tree_id": "013bf76937e440da65199134199dc5410e656337",
          "url": "https://github.com/GalacticDynamics/jaxmore/commit/524d2455ff761ded9e18f88cd252d4bbf4b17f6c"
        },
        "date": 1788796351377,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop",
            "value": 26.350737846034498,
            "unit": "iter/sec",
            "range": "stddev: 0.0014403797740439215",
            "extra": "mean: 37.949601481481444 msec\nrounds: 27"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop_jit",
            "value": 112250.57250432401,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019062137435266648",
            "extra": "mean: 8.908640532425606 usec\nrounds: 25318"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_closure",
            "value": 852.9927520336533,
            "unit": "iter/sec",
            "range": "stddev: 0.0001620670693331883",
            "extra": "mean: 1.172342903988177 msec\nrounds: 677"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_carry",
            "value": 815.7263336037862,
            "unit": "iter/sec",
            "range": "stddev: 0.0002159063176895321",
            "extra": "mean: 1.225901333333342 msec\nrounds: 666"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_via_step_kw",
            "value": 831.1988025847501,
            "unit": "iter/sec",
            "range": "stddev: 0.0002046475429437952",
            "extra": "mean: 1.2030816176471077 msec\nrounds: 748"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_empty_batch_skipping",
            "value": 1356.5417002455458,
            "unit": "iter/sec",
            "range": "stddev: 0.00011355129884063858",
            "extra": "mean: 737.1686397985342 usec\nrounds: 1191"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_single_positional",
            "value": 125467.18164648332,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011457873934493868",
            "extra": "mean: 7.9702117069753164 usec\nrounds: 56035"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_two_positionals",
            "value": 122725.15376245773,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011145188656887104",
            "extra": "mean: 8.148288833563518 usec\nrounds: 63476"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_with_kwonly",
            "value": 119999.30985244484,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013329232805362595",
            "extra": "mean: 8.333381260522525 usec\nrounds: 64132"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_outs_only",
            "value": 246326.0339117544,
            "unit": "iter/sec",
            "range": "stddev: 8.647345759560006e-7",
            "extra": "mean: 4.059660215851351 usec\nrounds: 107945"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_bind_free_pos_only",
            "value": 127816.0012062667,
            "unit": "iter/sec",
            "range": "stddev: 0.000001006131488251766",
            "extra": "mean: 7.823746561952143 usec\nrounds: 66971"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_varargs_bind_free",
            "value": 122626.45638481168,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011286800115484275",
            "extra": "mean: 8.15484708179057 usec\nrounds: 63001"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_pos_only_default_omitted",
            "value": 87773.95583509062,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017899015445628637",
            "extra": "mean: 11.392901122957205 usec\nrounds: 49334"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_static_path",
            "value": 2604.6307790636256,
            "unit": "iter/sec",
            "range": "stddev: 0.000042282957619620265",
            "extra": "mean: 383.9315760368553 usec\nrounds: 1302"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_kw_path",
            "value": 2088.3851814156455,
            "unit": "iter/sec",
            "range": "stddev: 0.00003285903641731136",
            "extra": "mean: 478.8388698114272 usec\nrounds: 1060"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_general_path",
            "value": 2327.269182248694,
            "unit": "iter/sec",
            "range": "stddev: 0.00009260663585056288",
            "extra": "mean: 429.6881545235617 usec\nrounds: 1249"
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
          "id": "8feff2bdd0d99b0161082174615f693cfad976b3",
          "message": "ci(prek): protect main and versions/ branches from direct commits (#26)\n\n* ci(prek): protect main and versions/ branches from direct commits\n* ci: explicitly set always_run: true on no-commit-to-branch\n* fix(ci): skip no-commit-to-branch in the full-suite CI run\n* fix(ci): don't clobber an existing SKIP when skipping no-commit-to-branch\n* docs(nox): clarify the no-commit-to-branch skip isn't CI-specific\n* docs: disambiguate \"never fires on push\" from the workflow's own push trigger\n\nCo-authored-by: Claude Sonnet 5 <noreply@anthropic.com>",
          "timestamp": "2026-09-15T16:52:13-04:00",
          "tree_id": "224082ffcf45255b36c20eb9703296905e675614",
          "url": "https://github.com/GalacticDynamics/jaxmore/commit/8feff2bdd0d99b0161082174615f693cfad976b3"
        },
        "date": 1789505818371,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop",
            "value": 26.259063153792034,
            "unit": "iter/sec",
            "range": "stddev: 0.003667573197939468",
            "extra": "mean: 38.08208975862078 msec\nrounds: 29"
          },
          {
            "name": "tests/benchmark/test_bounded_while_loop.py::test_bench_scalar_loop_jit",
            "value": 102352.91886817101,
            "unit": "iter/sec",
            "range": "stddev: 0.000002224126657448678",
            "extra": "mean: 9.770117071971193 usec\nrounds: 28333"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_closure",
            "value": 791.1906632130682,
            "unit": "iter/sec",
            "range": "stddev: 0.00009761980400927206",
            "extra": "mean: 1.2639178474869077 msec\nrounds: 577"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_in_carry",
            "value": 783.5678010235083,
            "unit": "iter/sec",
            "range": "stddev: 0.0001355157990974899",
            "extra": "mean: 1.2762137478004898 msec\nrounds: 682"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_optimizer_via_step_kw",
            "value": 781.7962026841336,
            "unit": "iter/sec",
            "range": "stddev: 0.000082984746605612",
            "extra": "mean: 1.279105726744015 msec\nrounds: 688"
          },
          {
            "name": "tests/benchmark/test_nn.py::test_bench_empty_batch_skipping",
            "value": 1191.4276790398296,
            "unit": "iter/sec",
            "range": "stddev: 0.000041709750131117266",
            "extra": "mean: 839.3291658339674 usec\nrounds: 1001"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_single_positional",
            "value": 119574.54540348084,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013944895024351244",
            "extra": "mean: 8.362983916231471 usec\nrounds: 52040"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_two_positionals",
            "value": 116703.7428246956,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013307330895827526",
            "extra": "mean: 8.568705474186306 usec\nrounds: 57068"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_fast_path_with_kwonly",
            "value": 114518.86416562958,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020400900332943066",
            "extra": "mean: 8.732185804372735 usec\nrounds: 58201"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_outs_only",
            "value": 240471.62546372108,
            "unit": "iter/sec",
            "range": "stddev: 7.962312688496712e-7",
            "extra": "mean: 4.158494783206203 usec\nrounds: 108496"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_bind_free_pos_only",
            "value": 122882.78697631203,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012185794897406455",
            "extra": "mean: 8.137836263371605 usec\nrounds: 60750"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_varargs_bind_free",
            "value": 118902.06800640159,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012327692746750853",
            "extra": "mean: 8.410282653335859 usec\nrounds: 57965"
          },
          {
            "name": "tests/benchmark/test_structured.py::test_bench_pos_only_default_omitted",
            "value": 83513.37039671074,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015844216538820746",
            "extra": "mean: 11.974130552386207 usec\nrounds: 43247"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_static_path",
            "value": 2247.9638634152375,
            "unit": "iter/sec",
            "range": "stddev: 0.000024239942493949794",
            "extra": "mean: 444.8470085638929 usec\nrounds: 1518"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_kw_path",
            "value": 1776.578914485045,
            "unit": "iter/sec",
            "range": "stddev: 0.00002994708199890324",
            "extra": "mean: 562.8795838150863 usec\nrounds: 1211"
          },
          {
            "name": "tests/benchmark/test_vmap.py::test_bench_general_path",
            "value": 1926.2101795299293,
            "unit": "iter/sec",
            "range": "stddev: 0.000037103748772505215",
            "extra": "mean: 519.1541456000607 usec\nrounds: 1250"
          }
        ]
      }
    ]
  }
}