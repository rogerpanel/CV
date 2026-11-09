# Independent detectors

## In distribution: test AUROC (macro-F1); fine-tuned: mean over seeds

| case | lexical | protectai | piguard | llmjudge | deberta_ft |
|---|---|---|---|---|---|
| a2a_sandbox | 0.995 (0.955) | 0.532 (0.559) | 0.571 (0.531) | 0.551 (0.550) | 0.897 (0.900) |
| agentdojo | 1.000 (1.000) | 0.704 (0.532) | 0.678 (0.629) | 0.895 (0.730) | 1.000 (0.962) |
| injecagent | 1.000 (1.000) | 0.168 (0.496) | 0.500 (0.496) | 0.548 (0.507) | 0.931 (0.785) |
| mcp_safetybench | 0.935 (0.822) | 0.462 (0.459) | 0.507 (0.508) | 0.635 (0.587) | 0.490 (0.454) |
| mcptox | 0.917 (0.899) | 0.691 (0.660) | 0.510 (0.473) | 0.971 (0.871) | 0.988 (0.958) |

## Held-out templates: mean test AUROC over folds

| case | lexical | protectai | piguard | llmjudge | deberta_ft |
|---|---|---|---|---|---|
| a2a_sandbox | 0.606 (0.455) | 0.564 (0.443) | 0.577 (0.430) | 0.617 (0.380) | 0.511 (0.348) |
| agentdojo | 0.981 (0.908) | 0.742 (0.674) | 0.592 (0.497) | 0.902 (0.782) | 0.883 (0.759) |
| injecagent | 1.000 (0.955) | 0.171 (0.483) | 0.500 (0.483) | 0.549 (0.583) | 0.774 (0.615) |
| mcp_safetybench | 0.670 (0.528) | 0.437 (0.398) | 0.502 (0.468) | 0.636 (0.508) | 0.564 (0.493) |
| mcptox | 0.870 (0.675) | 0.695 (0.620) | 0.508 (0.479) | 0.965 (0.900) | 0.976 (0.970) |

## Cross-benchmark transfer: test AUROC (train on source, test on target)

| case | lexical | protectai | piguard | llmjudge | deberta_ft |
|---|---|---|---|---|---|
| agentdojo from_injecagent | 0.341 (0.464) | 0.704 (0.464) | 0.678 (0.464) | 0.895 (0.533) | 0.631 (0.581) |
| injecagent from_agentdojo | 0.471 (0.016) | 0.168 (0.123) | 0.500 (0.413) | 0.548 (0.417) | 0.409 (0.016) |
| mcp_safetybench from_mcptox | 0.757 (0.520) | 0.462 (0.372) | 0.507 (0.394) | 0.635 (0.574) | 0.595 (0.499) |
| mcptox from_mcp_safetybench | 0.792 (0.582) | 0.691 (0.569) | 0.510 (0.450) | 0.971 (0.871) | 0.513 (0.401) |
