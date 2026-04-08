#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from scapy.all import DNS, DNSQR, IP, IPv6, Raw, TCP, UDP, get_if_addr, sniff  # pyright: ignore[reportMissingImports]


DEFAULT_TARGET_CLASSES = [
	"openlive",
	"live",
	"message",
	"short_video",
	"video",
	"meeting",
	"phone_game",
	"cloud_game",
]


DEFAULT_CLASS_KEYWORDS: dict[str, list[str]] = {
	"meeting": ["zoom", "meet", "teams", "webex", "voov"],
	"message": ["whatsapp", "telegram", "signal", "wechat", "qq"],
	"short_video": ["tiktok", "douyin", "kuaishou", "snssdk"],
	"video": ["youtube", "netflix", "bilibili", "iqiyi", "youku", "v.qq"],
	"live": ["twitch", "huya", "douyu", "live"],
	"openlive": ["obs", "rtmp", "livepush", "live-push"],
	"phone_game": ["game", "supercell", "riotgames", "miHoYo", "steam"],
	"cloud_game": ["xboxcloud", "geforcenow", "stadia", "boosteroid"],
}


HTTP_HOST_PATTERN = re.compile(rb"\r\nHost:\s*([^\r\n]+)", flags=re.IGNORECASE)
HTTP_UA_PATTERN = re.compile(rb"\r\nUser-Agent:\s*([^\r\n]+)", flags=re.IGNORECASE)
TLS_SNI_PATTERN = re.compile(rb"([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}")


@dataclass(frozen=True)
class FlowKey:
	proto: str
	a_ip: str
	a_port: int
	b_ip: str
	b_port: int


@dataclass
class FlowRecord:
	flow_id: int
	key: FlowKey
	start_ts: float
	first_seen: float
	last_seen: float
	packets: list[tuple[int, int, int]] = field(default_factory=list)
	uplink_packets: int = 0
	downlink_packets: int = 0
	uplink_bytes: int = 0
	downlink_bytes: int = 0
	dns_queries: set[str] = field(default_factory=set)
	http_hosts: set[str] = field(default_factory=set)
	user_agents: set[str] = field(default_factory=set)
	tls_snis: set[str] = field(default_factory=set)


def canonical_flow_key(proto: str, src_ip: str, src_port: int, dst_ip: str, dst_port: int) -> FlowKey:
	left = (src_ip, src_port)
	right = (dst_ip, dst_port)
	if left <= right:
		return FlowKey(proto=proto, a_ip=src_ip, a_port=src_port, b_ip=dst_ip, b_port=dst_port)
	return FlowKey(proto=proto, a_ip=dst_ip, a_port=dst_port, b_ip=src_ip, b_port=src_port)


def decode_text(raw: bytes) -> str:
	return raw.decode("utf-8", errors="ignore").strip().lower()


def extract_http_fields(payload: bytes) -> tuple[str | None, str | None]:
	host = None
	ua = None
	host_m = HTTP_HOST_PATTERN.search(payload)
	ua_m = HTTP_UA_PATTERN.search(payload)
	if host_m:
		host = decode_text(host_m.group(1))
	if ua_m:
		ua = decode_text(ua_m.group(1))
	return host, ua


def extract_possible_sni(payload: bytes) -> str | None:
	# Lightweight TLS SNI heuristic from raw payload. Useful for capture labeling,
	# but not guaranteed to parse every ClientHello variant.
	matches = TLS_SNI_PATTERN.findall(payload)
	if not matches:
		return None
	candidate = max(matches, key=len)
	text = decode_text(candidate)
	if len(text) < 4 or "." not in text:
		return None
	return text


def infer_candidate_labels(
	dns_queries: set[str],
	tls_snis: set[str],
	http_hosts: set[str],
	target_classes: list[str],
	class_keywords: dict[str, list[str]],
) -> list[str]:
	matched: list[str] = []
	haystack = " ".join(sorted(dns_queries | tls_snis | http_hosts))
	for cls in target_classes:
		keywords = class_keywords.get(cls, [])
		if any(kw.lower() in haystack for kw in keywords):
			matched.append(cls)
	return matched


def packet_5tuple(pkt) -> tuple[str, str, int, str, int] | None:
	ip_layer = None
	if IP in pkt:
		ip_layer = pkt[IP]
	elif IPv6 in pkt:
		ip_layer = pkt[IPv6]
	if ip_layer is None:
		return None

	src_ip = ip_layer.src
	dst_ip = ip_layer.dst

	if TCP in pkt:
		proto = "tcp"
		src_port = int(pkt[TCP].sport)
		dst_port = int(pkt[TCP].dport)
	elif UDP in pkt:
		proto = "udp"
		src_port = int(pkt[UDP].sport)
		dst_port = int(pkt[UDP].dport)
	else:
		return None
	return proto, src_ip, src_port, dst_ip, dst_port


def now_stamp() -> str:
	return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Collect packet sequence data by flow using Scapy.")
	parser.add_argument("--iface", type=str, default="any", help="Capture interface, e.g. eth0/wlan0/any")
	parser.add_argument("--duration", type=float, default=60.0, help="Capture duration in seconds.")
	parser.add_argument(
		"--idle-timeout",
		type=float,
		default=5.0,
		help="Flush a flow after this many seconds without new packets.",
	)
	parser.add_argument(
		"--min-packets",
		type=int,
		default=6,
		help="Minimum packets per flow to be persisted.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("datasets/raw/huawei"),
		help="Directory for output CSV/JSON files.",
	)
	parser.add_argument(
		"--capture-label",
		type=str,
		default="unknown",
		help="Output packet filename suffix, matching Huawei naming convention.",
	)
	parser.add_argument(
		"--bpf-filter",
		type=str,
		default="",
		help="Optional BPF filter, e.g. 'tcp or udp'.",
	)
	parser.add_argument(
		"--target-classes",
		type=str,
		default=",".join(DEFAULT_TARGET_CLASSES),
		help="Comma-separated target classes for candidate matching.",
	)
	parser.add_argument(
		"--keywords-json",
		type=Path,
		default=None,
		help="Optional JSON file mapping class name to keyword list.",
	)
	return parser.parse_args()


def load_class_keywords(path: Path | None) -> dict[str, list[str]]:
	if path is None:
		return DEFAULT_CLASS_KEYWORDS
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		raise ValueError("keywords JSON must be an object: class -> [keywords]")
	normalized: dict[str, list[str]] = {}
	for cls, words in data.items():
		if not isinstance(cls, str) or not isinstance(words, list):
			raise ValueError("keywords JSON entries must be class(string) -> list[string]")
		normalized[cls] = [str(w).lower() for w in words]
	return normalized


def main() -> None:
	args = parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)

	target_classes = [x.strip() for x in args.target_classes.split(",") if x.strip()]
	if not target_classes:
		raise ValueError("--target-classes cannot be empty")

	class_keywords = load_class_keywords(args.keywords_json)

	local_ips: set[str] = set()
	try:
		iface_ip = get_if_addr(args.iface)
		if iface_ip:
			local_ips.add(iface_ip)
	except Exception:
		pass
	local_ips.update({"127.0.0.1", "::1"})

	active_flows: dict[FlowKey, FlowRecord] = {}
	completed_flows: list[FlowRecord] = []
	next_flow_id = 1

	capture_start = time.time()

	def flush_idle(force: bool = False) -> None:
		now_t = time.time()
		expired: list[FlowKey] = []
		for key, record in active_flows.items():
			if force or (now_t - record.last_seen >= args.idle_timeout):
				expired.append(key)
		for key in expired:
			completed_flows.append(active_flows.pop(key))

	def on_packet(pkt) -> None:
		nonlocal next_flow_id
		parsed = packet_5tuple(pkt)
		if parsed is None:
			return
		proto, src_ip, src_port, dst_ip, dst_port = parsed
		key = canonical_flow_key(proto, src_ip, src_port, dst_ip, dst_port)

		timestamp = float(getattr(pkt, "time", time.time()))
		if key not in active_flows:
			active_flows[key] = FlowRecord(
				flow_id=next_flow_id,
				key=key,
				start_ts=capture_start,
				first_seen=timestamp,
				last_seen=timestamp,
			)
			next_flow_id += 1
		flow = active_flows[key]

		arrive_ms = int(round((timestamp - flow.first_seen) * 1000.0))
		direction = 1 if src_ip in local_ips else 0
		pkt_len = int(len(pkt))
		flow.packets.append((arrive_ms, direction, pkt_len))

		if direction == 1:
			flow.uplink_packets += 1
			flow.uplink_bytes += pkt_len
		else:
			flow.downlink_packets += 1
			flow.downlink_bytes += pkt_len

		flow.last_seen = timestamp

		if DNS in pkt and pkt[DNS].qd is not None and DNSQR in pkt:
			try:
				qname = pkt[DNSQR].qname
				if isinstance(qname, bytes):
					qname_text = decode_text(qname).strip(".")
				else:
					qname_text = str(qname).lower().strip(".")
				if qname_text:
					flow.dns_queries.add(qname_text)
			except Exception:
				pass

		if Raw in pkt:
			payload = bytes(pkt[Raw].load)
			host, ua = extract_http_fields(payload)
			if host:
				flow.http_hosts.add(host)
			if ua:
				flow.user_agents.add(ua)

			sni = extract_possible_sni(payload)
			if sni:
				flow.tls_snis.add(sni)

		flush_idle(force=False)

	sniff_kwargs = {
		"iface": args.iface,
		"prn": on_packet,
		"store": False,
		"timeout": args.duration,
	}
	if args.bpf_filter:
		sniff_kwargs["filter"] = args.bpf_filter

	print(f"[collector] start capture iface={args.iface}, duration={args.duration}s")
	sniff(**sniff_kwargs)
	flush_idle(force=True)

	packet_rows: list[dict[str, int]] = []
	flow_rows: list[dict[str, object]] = []

	for flow in completed_flows:
		if len(flow.packets) < args.min_packets:
			continue

		for arrive_ms, direction, pkt_len in flow.packets:
			packet_rows.append(
				{
					"flow_id": flow.flow_id,
					"arrive_time": arrive_ms,
					"direction": direction,
					"pkt_len": pkt_len,
				}
			)

		candidates = infer_candidate_labels(
			dns_queries=flow.dns_queries,
			tls_snis=flow.tls_snis,
			http_hosts=flow.http_hosts,
			target_classes=target_classes,
			class_keywords=class_keywords,
		)

		flow_rows.append(
			{
				"flow_id": flow.flow_id,
				"proto": flow.key.proto,
				"src_ip": flow.key.a_ip,
				"src_port": flow.key.a_port,
				"dst_ip": flow.key.b_ip,
				"dst_port": flow.key.b_port,
				"first_seen_iso": datetime.fromtimestamp(flow.first_seen).isoformat(),
				"last_seen_iso": datetime.fromtimestamp(flow.last_seen).isoformat(),
				"duration_ms": int(round((flow.last_seen - flow.first_seen) * 1000.0)),
				"packet_count": len(flow.packets),
				"uplink_packets": flow.uplink_packets,
				"downlink_packets": flow.downlink_packets,
				"uplink_bytes": flow.uplink_bytes,
				"downlink_bytes": flow.downlink_bytes,
				"dns_queries": "|".join(sorted(flow.dns_queries)),
				"tls_sni": "|".join(sorted(flow.tls_snis)),
				"http_host": "|".join(sorted(flow.http_hosts)),
				"user_agent": "|".join(sorted(flow.user_agents)),
				"candidate_labels": "|".join(candidates),
				"is_target_candidate": int(bool(candidates)),
			}
		)

	stamp = now_stamp()
	packet_path = (
		args.output_dir / f"network_traffic_classfication_packet_sequence-{args.capture_label}_{stamp}.csv"
	)
	meta_path = args.output_dir / f"network_traffic_classfication_flow_metadata-{args.capture_label}_{stamp}.csv"
	summary_path = args.output_dir / f"network_traffic_classfication_collection_summary-{args.capture_label}_{stamp}.json"

	with packet_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["flow_id", "arrive_time", "direction", "pkt_len"])
		writer.writeheader()
		writer.writerows(packet_rows)

	meta_fields = [
		"flow_id",
		"proto",
		"src_ip",
		"src_port",
		"dst_ip",
		"dst_port",
		"first_seen_iso",
		"last_seen_iso",
		"duration_ms",
		"packet_count",
		"uplink_packets",
		"downlink_packets",
		"uplink_bytes",
		"downlink_bytes",
		"dns_queries",
		"tls_sni",
		"http_host",
		"user_agent",
		"candidate_labels",
		"is_target_candidate",
	]
	with meta_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=meta_fields)
		writer.writeheader()
		writer.writerows(flow_rows)

	summary = {
		"capture_started_at": datetime.fromtimestamp(capture_start).isoformat(),
		"capture_duration_sec": args.duration,
		"iface": args.iface,
		"bpf_filter": args.bpf_filter,
		"target_classes": target_classes,
		"packet_csv": str(packet_path),
		"flow_metadata_csv": str(meta_path),
		"total_flows_captured": len(completed_flows),
		"total_flows_saved": len(flow_rows),
		"total_packets_saved": len(packet_rows),
		"candidate_flows": sum(int(row["is_target_candidate"]) for row in flow_rows),
	}
	with summary_path.open("w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	print(f"[collector] packet sequence saved: {packet_path}")
	print(f"[collector] flow metadata saved: {meta_path}")
	print(f"[collector] summary saved: {summary_path}")
	print(
		"[collector] stats "
		f"flows_captured={summary['total_flows_captured']} "
		f"flows_saved={summary['total_flows_saved']} "
		f"packets_saved={summary['total_packets_saved']}"
	)


if __name__ == "__main__":
	main()
