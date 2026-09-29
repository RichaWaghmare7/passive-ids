"""Top-level orchestrator -- wires every tier together in order.

This module intentionally contains no detection logic itself. It only
sequences calls into the tier-specific packages, so the flow here
should read as a direct match to the architecture diagrams: capture ->
flow_state -> features -> emission -> models -> calibration ->
alerting -> application / verdicts -> feedback.

All six models are now wired in (see on_feature_vector_ready and the
DNS/TLS blocks in on_packet), but every one of them ships unfit/
untrained by default -- see each model's own module docstring. That
means on_packet running today produces real DetectionResult objects,
not stub failures, but every raw_score reaching self.detections is a
cold-start "untrained" result until something calls .fit()/.update()
on the relevant model with real data (eventually via feedback/,
still not built). calibration/ and alerting/ are also still TODOs, so
self.detections holds raw, per-model, uncalibrated scores -- the same
accumulator pattern as ja4_hits and pending_feature_vectors, not a
finished alert pipeline.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from passive_ids.capture.dns_parser import DNSParser
from passive_ids.capture.l3l4_parser import L3L4Parser
from passive_ids.capture.tls_parser import TLSParser, compute_ja4
from passive_ids.emission.triggers import EmissionScheduler, FlowTimeoutTrigger, TumblingWindowTrigger
from passive_ids.feedback.feature_vector_store import FeatureVectorStore
from passive_ids.flow_state.dns_aggregate import DNSAggregateTable
from passive_ids.flow_state.flow_key import FlowKey
from passive_ids.flow_state.flow_table import BoundedFlowTable
from passive_ids.flow_state.source_aggregate import SourceAggregateTable
from passive_ids.intel.ja4_lookup import JA4LookupTable
from passive_ids.models.c2_beaconing import C2BeaconingDetector
from passive_ids.models.ddos_hoeffding import DDoSHoeffdingDetector
from passive_ids.models.dga_cascade import DGACascadeDetector
from passive_ids.models.exfiltration import ExfiltrationDetector
from passive_ids.models.malware_encrypted import EncryptedMalwareDetector, build_feature_vector
from passive_ids.models.port_scan import PortScanDetector


class PassiveIDSPipeline:
    # Not real feature-engineering versioning (no code-level tracking
    # exists yet) -- a fixed placeholder version string, honestly
    # disclosed as such rather than implying a real versioning system.
    # Bump manually if the shape of any feature vector below changes,
    # so old verdicts don't silently join against a different schema.
    FEATURE_VECTOR_VERSION = "fv-1"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.parser = L3L4Parser()
        self.dns_parser = DNSParser()
        self.tls_parser = TLSParser()
        self.source_aggregate = SourceAggregateTable()
        self.dns_aggregate = DNSAggregateTable()
        self.ja4_lookup = JA4LookupTable()
        self.ja4_hits: list[dict[str, Any]] = []  # pending alerting/ existing --
        # a simple accumulator for now, same spirit as confirmed flow
        # entries existing before models/ could consume them.
        self.flow_table = BoundedFlowTable()
        self.emission_scheduler = EmissionScheduler(
            FlowTimeoutTrigger(self.flow_table.confirmed),
            TumblingWindowTrigger(self.source_aggregate, self.dns_aggregate),
        )
        self.feature_vector_store = FeatureVectorStore()  # the join key
        # feedback/retrain.py needs: verdicts reference a flow_id, and
        # this is where the feature vector that flow_id actually
        # scored gets persisted, so a later verdict can be turned into
        # a real (X, y) training pair. Every detection point below
        # generates one id and records into this store before scoring.

        # All six detectors. Each ships unfit/untrained -- see their
        # own module docstrings. Calling .fit()/.update() on any of
        # these (with real data, once available) is what turns
        # dispatch below from cold-start 0.0 scores into real signal;
        # nothing in pipeline.py needs to change when that happens.
        self.ddos_model = DDoSHoeffdingDetector()
        self.c2_model = C2BeaconingDetector()
        self.dga_model = DGACascadeDetector()
        self.malware_model = EncryptedMalwareDetector()
        self.port_scan_model = PortScanDetector()
        self.exfil_model = ExfiltrationDetector()

        self.detections: list[dict[str, Any]] = []  # {"timestamp", "context", "detection"}
        self.pending_feature_vectors: list[dict[str, Any]] = []  # vectors whose
        # shape doesn't match any dispatch case yet -- currently just
        # the DNS aggregate's tumbling-window vectors (nxdomain_rate,
        # running domain-entropy stats): a genuine, disclosed gap, not
        # an oversight. The per-domain DGA cascade below is a
        # different signal (scores one domain string at a time, off
        # DNS queries directly) from what the DNS aggregate snapshot
        # would feed a model -- no model consumes THAT aggregate yet.

    def on_packet(self, raw_packet: bytes) -> None:
        """Entry point for one captured packet. Never blocks on I/O back
        to the network -- this pipeline is read-only by construction.

        Both lanes now run from the same parsed packet: the always-on
        aggregate (unaffected by flow-table capacity) and the bounded
        main lane (provisional -> promotion gate -> capacity check).
        """
        # 1. capture: parse L3/L4 fields (DNS/TLS parsers not yet built)
        parsed = self.parser.parse(raw_packet)
        if parsed is None:
            return  # not IPv4, or unrecognized framing -- correctly dropped, not an error

        timestamp = time.time()

        # 2a. flow_state: always-on source aggregate (unbounded lane)
        self.source_aggregate.update(
            source_ip=parsed.src_ip,
            dest_ip=parsed.dst_ip,
            dest_port=parsed.dst_port or 0,
            packet_size=parsed.packet_size,
            is_syn=parsed.syn,
            protocol=parsed.protocol,
        )

        # 2b. flow_state: provisional -> promotion gate -> capacity check
        #     (bounded lane). Returns None if still provisional or
        #     degraded -- both expected, not errors.
        key = FlowKey(
            src_ip=parsed.src_ip,
            dst_ip=parsed.dst_ip,
            src_port=parsed.src_port or 0,
            dst_port=parsed.dst_port or 0,
            protocol=parsed.protocol,
        )
        self.flow_table.observe_packet(
            key, timestamp=timestamp, size=parsed.packet_size, is_syn=parsed.syn
        )

        # 2c. capture + flow_state: DNS-specific aggregate, plus
        #     per-domain DGA scoring for actual queries (not
        #     responses -- a response repeats the same question, and
        #     scoring it twice would double-count the same domain).
        dns_msg = self.dns_parser.parse(raw_packet)
        if dns_msg is not None:
            self.dns_aggregate.observe(parsed.src_ip, parsed.dst_ip, dns_msg)
            if not dns_msg.is_response:
                for question in dns_msg.questions:
                    dga_fv = {"domain": question.qname}
                    flow_id = uuid.uuid4().hex[:12]
                    self.feature_vector_store.record(flow_id, self.FEATURE_VECTOR_VERSION, dga_fv)
                    result = self.dga_model.score(dga_fv)
                    self.detections.append(
                        {
                            "timestamp": timestamp,
                            "context": {
                                "src_ip": parsed.src_ip,
                                "domain": question.qname,
                                "verdict_flow_id": flow_id,
                            },
                            "detection": result,
                        }
                    )

        # 2d. capture + intel + models: TLS ClientHello -> JA4 ->
        #     known-bad lookup -> structural classifier. The lookup
        #     result feeds the classifier as one input feature
        #     (build_feature_vector) rather than being a separate
        #     parallel signal -- see models/malware_encrypted.py.
        client_hello = self.tls_parser.parse(raw_packet)
        if client_hello is not None:
            ja4 = compute_ja4(client_hello)
            match = self.ja4_lookup.lookup(ja4)
            if match is not None:
                self.ja4_hits.append(
                    {
                        "timestamp": timestamp,
                        "src_ip": parsed.src_ip,
                        "dst_ip": parsed.dst_ip,
                        "ja4": ja4,
                        "malware_family": match.malware_family,
                        "confidence": match.confidence,
                    }
                )
            malware_fv = build_feature_vector(client_hello, ja4_match=match)
            flow_id = uuid.uuid4().hex[:12]
            self.feature_vector_store.record(flow_id, self.FEATURE_VECTOR_VERSION, malware_fv)
            result = self.malware_model.score(malware_fv)
            self.detections.append(
                {
                    "timestamp": timestamp,
                    "context": {
                        "src_ip": parsed.src_ip,
                        "dst_ip": parsed.dst_ip,
                        "ja4": ja4,
                        "verdict_flow_id": flow_id,
                    },
                    "detection": result,
                }
            )

        # 3. features: Welford IAT tracking is already live inside
        #    ConfirmedFlowEntry.iat_tracker -- see flow_state/
        #    confirmed_table.py.

        # 4. emission -> 5. models: flow-timeout / 1s tumbling-window
        #    triggers hand emitted vectors to on_feature_vector_ready,
        #    which dispatches by vector shape (see that method).
        for vector in self.emission_scheduler.maybe_emit(timestamp):
            self.on_feature_vector_ready(vector)
        return

    def on_feature_vector_ready(self, feature_vector: dict[str, Any]) -> None:
        """Dispatches an emitted feature vector to whichever
        detector(s) its shape matches. calibration/ and alerting/
        don't exist yet, so results land in self.detections as raw,
        per-model, uncalibrated DetectionResults with context -- not
        a finished multi-label alert.
        """
        aggregate_type = feature_vector.get("aggregate_type")
        emission_reason = feature_vector.get("emission_reason")

        if aggregate_type == "source":
            flow_id = uuid.uuid4().hex[:12]
            self.feature_vector_store.record(flow_id, self.FEATURE_VECTOR_VERSION, feature_vector)
            context = {"source_ip": feature_vector.get("source_ip"), "verdict_flow_id": flow_id}
            for model in (self.ddos_model, self.port_scan_model):
                self.detections.append(
                    {
                        "timestamp": time.time(),
                        "context": context,
                        "detection": model.score(feature_vector),
                    }
                )
        elif emission_reason == "flow_timeout":
            flow_id = uuid.uuid4().hex[:12]
            self.feature_vector_store.record(flow_id, self.FEATURE_VECTOR_VERSION, feature_vector)
            context = {"flow_key": feature_vector.get("flow_key"), "verdict_flow_id": flow_id}
            for model in (self.c2_model, self.exfil_model):
                self.detections.append(
                    {
                        "timestamp": time.time(),
                        "context": context,
                        "detection": model.score(feature_vector),
                    }
                )
        else:
            # e.g. aggregate_type == "dns" -- disclosed gap, see
            # __init__'s note on pending_feature_vectors.
            self.pending_feature_vectors.append(feature_vector)
