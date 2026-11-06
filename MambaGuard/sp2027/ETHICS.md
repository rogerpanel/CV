# Ethics Considerations --- MambaGuard

This section follows the IEEE S&P 2027 Call for Papers guidance on ethics
considerations for research that touches on offensive capability, live
systems, personal data, and responsible disclosure.

## Dual-use of the adaptive-attack suite

MambaGuard is a defence, but a fair evaluation of it required building
adaptive attackers against each of its components. These attackers live in
`mambaguard/attacks/adaptive/` and cover: (i) a PGD-style continuous
perturbation attack against the state-space detector; (ii) a GCG-style
discrete prompt-optimisation attack against the projector; (iii) an
LLM-driven semantic rewriting attack that paraphrases malicious payloads
to evade pattern matching; and (iv) a graph-edit evasion that modifies
tool-call graphs to defeat structural checks. Each of these is implemented
as a research scaffold: it operates on offline traces, uses locally stored
model weights where applicable, and does not ship a working exploit
against any production MCP, A2A, or agent server. We have deliberately
excluded (a) network-level payload delivery, (b) credential handling for
real agent platforms, and (c) end-to-end "plug and play" attack scripts.
The released code enables reproduction of the attack strength numbers in
the paper without enabling turnkey attack on a deployed system. Following
acceptance we will coordinate a responsible-disclosure window with the
maintainers of the libraries listed in Section 4 before any further
expansion of the attack suite is released.

## Live systems

All protocol traces used in this paper are either synthetic (generated
with open-source MCP/A2A SDKs against local, isolated server stubs) or
drawn from public benchmarks: MCPSecBench, MCP-SafetyBench, MCPTox,
A2ABreak, AgentDojo, and InjecAgent. No live third-party agent
deployment was targeted at any point during this work. We did not scan,
probe, or attack any production endpoint. Where our results refer to
"deployment" they refer to our own internal benchmark stack, and we did
not record or evaluate operator traffic from that stack in this paper.

## Personal and sensitive data

The benchmarks used contain only the content that their original authors
released. We added no personal identifiers, user accounts, or operator
traffic. Where a public benchmark includes synthetic names or addresses,
we treat those as data, not as records, and we have not attempted to
re-identify any underlying person. No participant data was collected from
humans for this work; no IRB review was therefore required under the
authors' institutional policy, and we confirm that policy covers this
class of work.

## Vulnerability disclosure

This submission reports no CVE identifiers. Where our analysis of a
public benchmark surfaced what we believe to be a previously
unreported weakness in a protocol library or agent framework, we
reached out privately to the relevant maintainers before preparing the
camera-ready, following a 90-day coordinated-disclosure window. Fixes,
mitigations, or acknowledgements will be cited in the camera-ready. We
will not publish proof-of-concept exploits for issues that remain
unpatched at publication.

## Reviewers and the artifact

The anonymised reviewer mirror contains the scaffolded attack code
described above. We ask reviewers to treat it as research artifact: run
it only inside the containerised evaluation harness provided, and do not
redirect it at any system other than the local benchmark server stubs we
ship. The licence under which the code is released (see `LICENSE` in the
anonymous mirror) is permissive for review, with a research-only clause
that will be revisited at camera-ready.
