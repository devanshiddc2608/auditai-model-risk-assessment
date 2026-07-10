# AuditAI — AI Decision Failure Audit System

**Live App:** [your-render-url-here]  
![AuditAI Demo](demo.gif)

## The Problem
Most ML projects report accuracy and stop there. But in high-stakes decisions — lending, hiring, insurance — the most dangerous failures aren't the low-confidence ones a human would catch. They're the ones the model got wrong while claiming near-certainty, because those are exactly the predictions nobody double-checks.

## What This Project Does
AuditAI trains a credit risk model, then systematically audits it — not for accuracy, but for **high-confidence failures**. It builds a failure taxonomy, quantifies the business cost of those failures, and deploys a live decision-support tool that warns reviewers in real time when a new case resembles a known past failure.

## Key Finding
- High-confidence (≥85%) prediction error rate: **3.10%**
- These high-confidence failures account for **48.5% of total error cost**
- Cost-optimal decision threshold: **0.86** (vs. naive default of 0.50)
- 25–40 age group is **1.5x overrepresented** in high-confidence failures

## What Makes This Different
1. **Failure-first framing** — failures are the primary object of study, not a footnote
2. **Business impact quantification** — every error is tied to an assumed cost, not just a metric
3. **Live risk-aware decisioning** — the deployed app doesn't just predict, it checks new cases against known failure patterns in real time

## Architecture