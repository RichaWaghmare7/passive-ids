<div align="center">

# 🛡️ Passive-IDS

### AI/ML-Powered Passive Intrusion Detection System for Critical Infrastructure

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-0074D9?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![SIH 2026](https://img.shields.io/badge/SIH-2026%20%7C%20PS%20145-red?style=for-the-badge)](.)

> **⚠️ Read-Only by Design:** This system passively monitors unidirectional IP traffic and never sends a single packet back. No active mitigation. No network interference. Pure intelligence.

</div>

---

## 📌 Table of Contents

- [What Is This?](#-what-is-this)
- [The Problem We're Solving](#-the-problem-were-solving)
- [How It Works — The Pipeline](#-how-it-works--the-pipeline)
- [The 6 Threat Classes Detected](#-the-6-threat-classes-detected)
- [System Architecture](#-system-architecture)
- [Key Design Decisions](#-key-design-decisions)
- [Project Structure](#-project-structure)
- [Dashboard & Application](#-dashboard--application)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Tech Stack](#-tech-stack)
- [Team & SIH Context](#-team--sih-context)

---

## 🔍 What Is This?

**Passive-IDS** is a multi-stage, AI/ML-powered **Intrusion Detection System** designed for **critical infrastructure** (power grids, water treatment plants, industrial control systems). It works by silently listening to a copy of network traffic — like a security camera that never touches what it's watching.

### Why "Passive"?

In critical infrastructure, **you cannot risk active probing or packet injection**. A false-positive firewall block on a power grid controller can be more catastrophic than the attack itself. Passive-IDS separates **detection from response** entirely:

```
Network Traffic ──► [Mirror/TAP Port] ──► Passive-IDS ──► Alerts ──► SOC Analysts ──► Human Decision
                                               ↑
                              (Never touches the original network)
```

---

## 🚨 The Problem We're Solving

Critical infrastructure networks are increasingly targeted by nation-state actors and ransomware groups. The challenge:

| Challenge | Why It's Hard |
|-----------|---------------|
| **Can't go offline for updates** | Power grids and water systems run 24/7 |
| **Legacy OT/SCADA protocols** | Old equipment never designed with security in mind |
| **High false-positive cost** | A wrong block can shut down essential services |
| **Multi-vector attacks** | Attackers combine DDoS + C2 + Exfiltration simultaneously |
| **Encrypted traffic is opaque** | TLS hides malware communication from signature-based IDS |

Passive-IDS tackles all of these by being **read-only, ML-driven, and multi-label** — it can flag DDoS AND C2 beaconing on the same flow at the same time.

---

## ⚙️ How It Works — The Pipeline

The system processes every packet through a strict, ordered pipeline. Data flows in **one direction only** — each tier feeds the next.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          PASSIVE-IDS DETECTION PIPELINE                             │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┤
│          │          │          │          │          │          │                  │
│  TIER 0  │  TIER 2  │  TIER 4  │ EMISSION │  TIER 5  │  TIER 6  │     TIER 7       │
│          │          │          │          │          │          │                  │
│ Capture  │  Flow    │ Feature  │ Timeout  │  6 ML    │  Calib-  │  Multi-Label     │
│  L3/L4   │  State   │  Extract │ Triggers │ Detectors│  ration  │    Alerting      │
│  DNS/TLS │  Tables  │  Welford │ Tumbling │          │ Isotonic │   JSON + SOC     │
│          │          │  Entropy │ Windows  │          │ per-class│   Dashboard      │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┘
```

### Step-by-Step Flow

**1. 📦 Packet Capture (Tier 0–1)**
Every raw packet is parsed for:
- **L3/L4 fields** — source/dest IP, ports, protocol, packet size, SYN flags
- **DNS messages** — queries (not responses, to avoid double-counting)
- **TLS ClientHello** — parsed to compute the **JA4 fingerprint** for encrypted traffic analysis

**2. 🗂️ Flow State (Tier 2–3) — The Dual-Lane Design**
This is the most critical tier. It runs **two independent lanes simultaneously**:

```
Every Packet
    │
    ├──► Always-On Source Aggregate (Lane 1) ──► HyperLogLog + Count-Min Sketch
    │         (Never full, never degrades)           (Feeds DDoS & Port Scan)
    │
    └──► Bounded Flow Table (Lane 2) ──► Provisional → Promotion Gate → Confirmed
              (Capacity-capped to prevent)        (Feeds C2 & Exfiltration)
              (detector resource exhaustion)
```

> **Why two lanes?** During a DDoS flood, the flow table is under maximum pressure — that's *exactly* when the DDoS detector needs the source aggregate most. Merging the lanes would blind the detector at the worst possible moment.

**3. 📊 Feature Extraction (Tier 4)**
Online statistical features computed per-flow with **zero memory growth**:
- Welford's online mean/variance for inter-arrival times (IAT)
- Shannon entropy of destination port distribution
- Count-Min Sketch for frequency estimation

**4. 🕐 Emission Triggers**
Feature vectors are emitted (sent to models) by two triggers:
- **Flow Timeout Trigger** — when a TCP flow goes idle (feeds C2 & Exfiltration detectors)
- **Tumbling Window Trigger** — every 1 second of source aggregate snapshot (feeds DDoS & Port Scan)

**5. 🤖 The 6 ML Detectors (Tier 5)**
See [next section](#-the-6-threat-classes-detected).

**6. 📐 Calibration (Tier 6)**
Six algorithm families produce six incompatible output scales. Isotonic regression maps every raw score to a consistent 0–1 probability, fit once per model instance.

**7. 🚨 Multi-Label Alerting (Tier 7)**
A single network flow can trigger multiple detectors simultaneously. The system emits **multi-label alert JSON** — never a single fused "risk score" that would hide compound attacks.

---

## 🎯 The 6 Threat Classes Detected

| # | Threat Class | Algorithm | Learning Mode | What It Catches |
|---|---|---|---|---|
| 1 | **DDoS** | Hoeffding Tree | 🟢 Online (live updates) | Volumetric floods, amplification attacks |
| 2 | **C2 Beaconing** | Isolation Forest | 🟠 Batch (retrain + canary) | Periodic callbacks to attacker-controlled servers |
| 3 | **DGA Detection** | Cascade (n-gram → entropy → LightGBM) | 🟠 Batch | Domain Generation Algorithm malware, DNS tunneling |
| 4 | **Port Scanning** | LightGBM | 🟠 Batch | Reconnaissance, service enumeration |
| 5 | **Data Exfiltration** | XGBoost | 🟠 Batch | Large outbound transfers, slow-drip exfil |
| 6 | **Encrypted Malware** | JA4 Lookup + Classifier | 🟠 Batch | Malware hiding in TLS, known-bad TLS fingerprints |

### 🟢 Online vs 🟠 Batch — Two Learning Cadences

This split is **deliberate and important**:

```
Online Learners (Hoeffding Tree)          Batch Models (IsoForest, LightGBM, XGBoost)
─────────────────────────────────         ──────────────────────────────────────────
✅ Update per-packet, no downtime          ✅ High accuracy on slower-evolving threats
✅ Adapts instantly to new DDoS patterns   ✅ Validated before deployment
❌ Less accurate on complex patterns       ❌ Replaced via canary deploy, never mutated
                                              while serving live traffic
```

---

## 🏗️ System Architecture

> 📷 **[Dashboard Screenshot Placeholder]**
> *Replace this with an actual screenshot of the SOC dashboard once deployed*
>
> ![Dashboard Screenshot](docs/images/dashboard_screenshot.png)

---

> 📷 **[Pipeline Architecture Diagram Placeholder]**
> *A visual flow diagram of the 8-tier detection pipeline*
>
> ![Architecture Diagram](docs/images/architecture_diagram.png)

---

> 📷 **[Alert Flow Diagram Placeholder]**
> *How a packet turns into a multi-label alert JSON*
>
> ![Alert Flow](docs/images/alert_flow.png)

---

## 🧠 Key Design Decisions

These are the "why" answers judges often ask about:

### 1. Why multi-label alerts instead of a single risk score?

A flow can simultaneously exhibit **C2 beaconing** (regular timing) AND **DNS tunneling** (DGA-like query names). A weighted sum into one number hides this. `Detection` objects are a list — and `overall_score` exists only as a **triage sort key**, never presented as a calibrated probability.

### 2. Why is calibration a separate tier?

An Isolation Forest anomaly score and a LightGBM sigmoid output are mathematically incomparable. Isotonic regression calibration maps each to a true 0–1 probability. This calibration is fit **per model instance**, not per threat class — because C2's Isolation Forest and Exfiltration's Isolation Forest are trained on entirely different feature distributions.

### 3. Why are thresholds `null` in config?

```yaml
# config/thresholds.yaml
thresholds:
  ddos: null       # ← honest placeholder
  c2_beaconing: null
  ...
```

Filling these with guessed numbers would make them look authoritative when they aren't. They need real PR curves from labeled data. `null` is the **honest answer** until `scripts/compute_pr_curves.py` runs on real traffic.

### 4. Why sample sub-threshold flows for SOC verdicts?

If analysts only review alerts that crossed a threshold, **false negatives are structurally invisible** — there's no ground truth for flows never shown to anyone. The verdict sampler distinguishes:
- `alerted` → precision signal
- `random` → unbiased recall signal  
- `boundary` → uncertainty-sampled (useful for retraining, but biased — kept separate)

### 5. Why JA4 fingerprinting for encrypted traffic?

TLS content is opaque, but the **structure** of a TLS ClientHello (cipher suites, extensions, their order) is highly distinctive per malware family. JA4 fingerprints map directly to known malware families even through TLS 1.3.

---

## 📁 Project Structure

```
passive-ids/
│
├── 📄 README.md                     ← You are here
├── 📄 requirements.txt              ← All dependencies
│
├── ⚙️  config/
│   ├── sla_matrix.yaml              ← Escalation tiers (CRITICAL/HIGH/MEDIUM)
│   └── thresholds.yaml              ← Per-class detection thresholds
│
├── 📚 docs/
│   └── architecture.md             ← Deep design rationale
│
├── 🔬 scripts/
│   ├── replay_pcap.py               ← Replay a PCAP file through the pipeline
│   ├── compute_pr_curves.py         ← Fit thresholds from labeled data
│   └── train_models.py              ← Batch model training entry point
│
├── 🔧 src/passive_ids/
│   ├── pipeline.py                  ← 🎯 Main orchestrator (start here)
│   │
│   ├── capture/                     ← Tier 0-1: Packet parsing
│   │   ├── l3l4_parser.py           ← IPv4/TCP/UDP field extraction
│   │   ├── dns_parser.py            ← DNS query/response parsing
│   │   └── tls_parser.py            ← TLS ClientHello + JA4 fingerprinting
│   │
│   ├── flow_state/                  ← Tier 2-3: Dual-lane flow tracking
│   │   ├── flow_table.py            ← Bounded confirmed flow table
│   │   ├── source_aggregate.py      ← Always-on HLL + Count-Min per source IP
│   │   └── dns_aggregate.py         ← NXDOMAIN rates, domain entropy
│   │
│   ├── features/                    ← Tier 4: Online feature computation
│   │
│   ├── emission/                    ← Timeout & tumbling-window triggers
│   │   └── triggers.py
│   │
│   ├── models/                      ← Tier 5: The 6 detectors
│   │   ├── base.py                  ← Shared DetectionResult interface
│   │   ├── ddos_hoeffding.py        ← DDoS via Hoeffding Tree (online)
│   │   ├── c2_beaconing.py          ← C2 via Isolation Forest (batch)
│   │   ├── dga_cascade.py           ← DGA via n-gram cascade (batch)
│   │   ├── port_scan.py             ← Port scan via LightGBM (batch)
│   │   ├── exfiltration.py          ← Exfil via XGBoost (batch)
│   │   └── malware_encrypted.py     ← Encrypted malware via JA4 + classifier
│   │
│   ├── calibration/                 ← Tier 6: Isotonic calibration per model
│   │
│   ├── alerting/                    ← Tier 7: Multi-label alert JSON
│   │
│   ├── intel/
│   │   └── ja4_lookup.py            ← Known-bad JA4 fingerprint database
│   │
│   ├── verdicts/                    ← SOC verdict schema + sampling
│   │
│   ├── feedback/                    ← Retrain → recalibrate → canary deploy
│   │   ├── retrain.py
│   │   └── feature_vector_store.py  ← Links verdicts back to training features
│   │
│   └── application/                 ← User-facing layer
│       ├── dashboard/               ← FastAPI + real-time alert dashboard
│       ├── case_management/         ← SOC analyst case workflow
│       └── siem_export.py           ← SIEM integration export
│
└── 🧪 tests/                        ← pytest test suite
```

---

## 📊 Dashboard & Application

> 📷 **[SOC Dashboard — Alert Table Placeholder]**
> *Real-time multi-label alert table with severity badges*
>
> ![Alert Table](docs/images/alert_table.png)

---

> 📷 **[Live Threat Timeline Placeholder]**
> *Timeline chart showing detection events across all 6 threat classes*
>
> ![Threat Timeline](docs/images/threat_timeline.png)

---

> 📷 **[Case Management UI Placeholder]**
> *SOC analyst verdict workflow — alerted / random / boundary sampling*
>
> ![Case Management](docs/images/case_management.png)

---

The application layer exposes:

| Component | Description | Tech |
|-----------|-------------|------|
| **SOC Dashboard** | Real-time alert feed, severity triage, flow details | FastAPI + WebSocket |
| **Case Management** | Analyst verdict collection with sampling labels | FastAPI REST |
| **SIEM Export** | Push alerts to Splunk / QRadar / Elastic SIEM | JSON over HTTP |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- (Optional) libpcap / Npcap for live capture

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/passive-ids.git
cd passive-ids

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the test suite
pytest tests/ -v
```

### Replay a PCAP File (Demo Mode)

```bash
# Feed a captured packet trace through the full pipeline
python scripts/replay_pcap.py --pcap path/to/capture.pcap --output alerts.json
```

### Run the SOC Dashboard Locally

```bash
uvicorn passive_ids.application.dashboard:app --reload --port 8000
# Open: http://localhost:8000
```

---

## ⚙️ Configuration

### SLA / Escalation Matrix (`config/sla_matrix.yaml`)

Defines how alerts are triaged and escalated:

| Severity | Notification | Escalates To |
|----------|-------------|--------------|
| `CRITICAL` | PagerDuty + Slack | Tier-2 Lead (paged) |
| `HIGH` | Slack + Dashboard | Tier-2 SOC |
| `MEDIUM` | Dashboard only | Tier-2 (backlog) |
| `SAMPLED` | None (best-effort) | Weekly batch review |

> **Note:** Response SLA values are `null` until a pilot deployment measures real alert arrival rates for Erlang-C staffing calculations.

### Detection Thresholds (`config/thresholds.yaml`)

Per-class precision/recall operating points. Currently `null` — populated by running `scripts/compute_pr_curves.py` against labeled traffic data.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Packet Parsing** | Scapy 2.5+ | L3/L4/DNS/TLS parsing (prototype) |
| **Online ML** | River 0.21+ | Hoeffding Tree for DDoS (live updates) |
| **Batch ML** | scikit-learn 1.4+ | Isolation Forest, Isotonic calibration |
| **Batch ML** | LightGBM 4.3+ | Port scan, DGA cascade |
| **Batch ML** | XGBoost 2.0+ | Exfiltration detection |
| **Cardinality Sketches** | datasketch (HyperLogLog) + mmh3 | Bounded-memory source aggregate |
| **Schemas** | Pydantic 2.6+ | Typed feature vectors, alert models |
| **Config** | PyYAML 6.0+ | Thresholds, SLA matrix |
| **API** | FastAPI 0.110+ | SOC dashboard, SIEM export |
| **Runtime** | Uvicorn 0.29+ | ASGI server |
| **Testing** | pytest 8.0+ | Unit and integration tests |

---

## 📈 Feedback & Continuous Learning Loop

```
SOC Analyst Verdicts
       │
       ▼
 Verdict Sampler ──► alerted (precision)
       │            ──► random   (recall, unbiased)
       │            ──► boundary (uncertainty, biased — kept separate)
       │
       ▼
 Feature Vector Store  ←── links verdict back to the scored features
       │
       ▼
 feedback/retrain.py   ──► retrain batch models
       │
       ▼
 feedback/recalibrate  ──► refit isotonic calibration
       │
       ▼
 Canary Deployment     ──► shadow traffic → shadow scoring → promote or rollback
```

Batch models are **never mutated while serving live traffic**. Updates only happen through this validated canary pipeline.

---

## 👥 Team & SIH Context

This project is built for **Smart India Hackathon 2026 — Problem Statement 145**.

> **PS 145:** Design and develop a passive intrusion detection system for critical infrastructure that can identify cyber threats in unidirectional, read-only network traffic without disrupting operational technology networks.

| | |
|---|---|
| **Track** | Cybersecurity |
| **Domain** | Critical Infrastructure Protection |
| **Approach** | Passive monitoring + Multi-class ML |
| **Stage** | Scaffold / Architecture (implementation in progress) |

---

[![SIH 2026](https://img.shields.io/badge/Smart%20India%20Hackathon-2026-orange?style=for-the-badge)](.)

</div>
