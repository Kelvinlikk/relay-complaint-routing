# Try the complaint review desk

**Start here:** open [field-walkthrough.html](field-walkthrough.html) in a browser. It works offline, without installing Python. Its six fictional complaints contain saved predictions from the frozen model. Choose **Card charge dispute**, then **Collection after payment**: both have very high scores, but only one destination passed the June audit. Try **Mortgage lien release** to see how the later, proposed hold for short messages changes the treatment of a case. The reviewer can always pick another desk.

The **Reviewer shortlist**, **Evidence**, and **Pilot plan** tabs show what was measured and what still needs a field trial. The June rule and the short-message safeguard are labeled separately. The shortlist figures use the entire expanded May cohort or the July cohort, as selected. A label mismatch is a disagreement with the published product, not a proven routing mistake.

**Score your own fictional example:** in an environment with the trained linear and CatBoost weights (from a completed refit or the team's private checkpoint), run the command below from the repository root and open http://127.0.0.1:8765.

```bash
python demo/desk.py --repo .
```

The local server recomputes scores; it does not log inputs, send complaints, or change customer records. Enter a company only when the incoming mailbox identifies the institution independently. The public source archive omits learned weights and vocabularies. An ordinary browser can replay the six examples in either edition.

**Scope:** people remain responsible for urgency, language, escalation and final routing decisions. A high score is not permission to transfer a case. Use only made-up text in the demo.
