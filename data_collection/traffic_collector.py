#!/usr/bin/env python3
# python.exe traffic_collector.py --iface WLAN --duration 3600 --capture-label video_client_play --bpf-filter "tcp or udp" --output-dir ..\datasets\raw\mydata\video
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Callable


SCAPY_CACHE_HOME = Path(__file__).resolve().parents[1] / ".cache"
SCAPY_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(SCAPY_CACHE_HOME))

from scapy.all import DNS, DNSQR, IP, IPv6, Raw, TCP, UDP, conf, get_if_addr, get_if_list, sniff  # pyright: ignore[reportMissingImports]
from scapy.interfaces import resolve_iface  # pyright: ignore[reportMissingImports]


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


@dataclass(frozen=True)
class CaptureConfig:
	iface: str = "any"
	duration: float = 60.0
	flush_interval_sec: float = 1.0
	idle_timeout: float = 5.0
	min_packets: int = 6
	output_dir: Path = Path("../datasets/raw/mydata")
	capture_label: str = "unknown"
	bpf_filter: str = ""
	target_classes: tuple[str, ...] = tuple(DEFAULT_TARGET_CLASSES)
	keywords_json: Path | None = None


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
		"--flush-interval-sec",
		type=float,
		default=1.0,
		help="Periodic flush interval in seconds for incremental updates.",
	)
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
		default=Path("../datasets/raw/mydata"),
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


def _emit(logger: Callable[[str], None] | None, message: str) -> None:
	if logger is not None:
		logger(message)
		return
	print(message)


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


def resolve_capture_interfaces(iface_arg: str | None) -> tuple[str | list[str] | None, list[str], str]:
	iface_name = (iface_arg or "").strip()
	if not iface_name:
		default_iface = str(conf.iface)
		return None, [default_iface], default_iface

	if iface_name.lower() == "any":
		available_ifaces = [name for name in get_if_list() if name]
		if not available_ifaces:
			raise ValueError("No capture interfaces are available on this host.")
		return available_ifaces, available_ifaces, "any"

	try:
		resolve_iface(iface_name)
	except ValueError as exc:
		available_ifaces = ", ".join(get_if_list()) or "<none>"
		raise ValueError(
			f"Interface '{iface_name}' not found. Available interfaces: {available_ifaces}. "
			"Use --iface any to capture on all interfaces."
		) from exc

	return iface_name, [iface_name], iface_name


def collect_local_ips(ifaces: list[str]) -> set[str]:
	local_ips = {"127.0.0.1", "::1"}
	for iface_name in ifaces:
		try:
			iface_ip = get_if_addr(iface_name)
		except Exception:
			continue
		if iface_ip and iface_ip != "0.0.0.0":
			local_ips.add(iface_ip)

	try:
		for family, _, _, _, sockaddr in socket.getaddrinfo(socket.gethostname(), None):
			if family not in (socket.AF_INET, socket.AF_INET6):
				continue
			address = sockaddr[0]
			if address:
				local_ips.add(address.split("%", maxsplit=1)[0])
	except socket.gaierror:
		pass

	return local_ips


def _run_capture_loop(
	*,
	duration: float,
	stop_event: Event | None,
	sniff_kwargs: dict[str, object],
	tick_interval: float,
	on_tick: Callable[[], None] | None,
) -> str:
	deadline = time.time() + max(duration, 0.0)
	stop_reason = "duration_elapsed"
	interval = max(tick_interval, 0.1)
	while True:
		if stop_event is not None and stop_event.is_set():
			stop_reason = "stop_requested"
			break
		remaining = deadline - time.time()
		if remaining <= 0.0:
			break
		sniff(timeout=min(interval, remaining), **sniff_kwargs)
		if on_tick is not None:
			on_tick()
	return stop_reason


def collect_traffic(
	config: CaptureConfig,
	*,
	stop_event: Event | None = None,
	logger: Callable[[str], None] | None = None,
	batch_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
	config.output_dir.mkdir(parents=True, exist_ok=True)

	target_classes = [x.strip() for x in config.target_classes if x.strip()]
	class_keywords = load_class_keywords(config.keywords_json) if target_classes else {}
	capture_iface, capture_ifaces, capture_iface_label = resolve_capture_interfaces(config.iface)

	local_ips = collect_local_ips(capture_ifaces)

	active_flows: dict[FlowKey, FlowRecord] = {}
	completed_flows: list[FlowRecord] = []
	pending_completed_flows: list[FlowRecord] = []
	next_flow_id = 1

	capture_start = time.time()

	def flush_idle(force: bool = False) -> None:
		now_t = time.time()
		expired: list[FlowKey] = []
		for key, record in active_flows.items():
			if force or (now_t - record.last_seen >= config.idle_timeout):
				expired.append(key)
		for key in expired:
			record = active_flows.pop(key)
			completed_flows.append(record)
			pending_completed_flows.append(record)

	def candidate_labels_for_flow(flow: FlowRecord) -> list[str]:
		if not target_classes:
			return []
		return infer_candidate_labels(
			dns_queries=flow.dns_queries,
			tls_snis=flow.tls_snis,
			http_hosts=flow.http_hosts,
			target_classes=target_classes,
			class_keywords=class_keywords,
		)

	def build_rows(flows: list[FlowRecord]) -> tuple[list[dict[str, object]], list[dict[str, int]], int]:
		packet_rows: list[dict[str, int]] = []
		flow_rows: list[dict[str, object]] = []
		candidate_count = 0
		for flow in flows:
			if len(flow.packets) < config.min_packets:
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

			candidates = candidate_labels_for_flow(flow)
			candidate_count += int(bool(candidates))
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
		return flow_rows, packet_rows, candidate_count

	def build_running_stats() -> tuple[int, int, int, int]:
		total_flows_captured = len(completed_flows) + len(active_flows)
		total_flows_saved = 0
		total_packets_saved = 0
		candidate_flows = 0
		for flow in [*completed_flows, *active_flows.values()]:
			if len(flow.packets) < config.min_packets:
				continue
			total_flows_saved += 1
			total_packets_saved += len(flow.packets)
			candidate_flows += int(bool(candidate_labels_for_flow(flow)))
		return total_flows_captured, total_flows_saved, total_packets_saved, candidate_flows

	def emit_batch() -> None:
		if batch_callback is None:
			pending_completed_flows.clear()
			return
		snapshot_flows = [
			flow
			for flow in [*pending_completed_flows, *active_flows.values()]
			if len(flow.packets) >= config.min_packets
		]
		pending_completed_flows.clear()
		if not snapshot_flows:
			return
		flow_rows, packet_rows, _ = build_rows(snapshot_flows)
		total_flows_captured, total_flows_saved, total_packets_saved, candidate_flows = build_running_stats()
		batch_callback(
			{
				"flow_rows_data": flow_rows,
				"packet_rows_data": packet_rows,
				"total_flows_captured": total_flows_captured,
				"total_flows_saved": total_flows_saved,
				"total_packets_saved": total_packets_saved,
				"candidate_flows": candidate_flows,
			}
		)

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
		direction = 0 if src_ip in local_ips else 1
		pkt_len = int(len(pkt))
		flow.packets.append((arrive_ms, direction, pkt_len))

		if direction == 0:
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

	sniff_kwargs: dict[str, object] = {
		"prn": on_packet,
		"store": False,
	}
	if capture_iface is not None:
		sniff_kwargs["iface"] = capture_iface
	if config.bpf_filter:
		sniff_kwargs["filter"] = config.bpf_filter

	if capture_iface_label == "any":
		iface_display = f"any ({', '.join(capture_ifaces)})"
	else:
		iface_display = capture_iface_label
	_emit(logger, f"[collector] start capture iface={iface_display}, duration={config.duration}s")
	stop_reason = _run_capture_loop(
		duration=config.duration,
		stop_event=stop_event,
		sniff_kwargs=sniff_kwargs,
		tick_interval=config.flush_interval_sec,
		on_tick=emit_batch,
	)
	flush_idle(force=True)
	emit_batch()

	flow_rows, packet_rows, candidate_flow_count = build_rows(completed_flows)

	stamp = now_stamp()
	packet_path = config.output_dir / f"network_traffic_classfication_packet_sequence-{config.capture_label}_{stamp}.csv"
	meta_path = config.output_dir / f"network_traffic_classfication_flow_metadata-{config.capture_label}_{stamp}.csv"
	summary_path = config.output_dir / f"network_traffic_classfication_collection_summary-{config.capture_label}_{stamp}.json"

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

	capture_end = time.time()
	summary = {
		"capture_started_at": datetime.fromtimestamp(capture_start).isoformat(),
		"capture_stopped_at": datetime.fromtimestamp(capture_end).isoformat(),
		"capture_duration_sec": round(capture_end - capture_start, 3),
		"configured_duration_sec": config.duration,
		"stop_reason": stop_reason,
		"iface": config.iface,
		"resolved_ifaces": capture_ifaces,
		"bpf_filter": config.bpf_filter,
		"target_classes": target_classes,
		"packet_csv": str(packet_path),
		"flow_metadata_csv": str(meta_path),
		"total_flows_captured": len(completed_flows),
		"total_flows_saved": len(flow_rows),
		"total_packets_saved": len(packet_rows),
		"candidate_flows": candidate_flow_count,
	}
	with summary_path.open("w", encoding="utf-8") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	summary["summary_json"] = str(summary_path)
	summary["flow_rows_data"] = flow_rows
	summary["packet_rows_data"] = packet_rows
	_emit(logger, f"[collector] packet sequence saved: {packet_path}")
	_emit(logger, f"[collector] flow metadata saved: {meta_path}")
	_emit(logger, f"[collector] summary saved: {summary_path}")
	_emit(
		logger,
		"[collector] stats "
		f"flows_captured={summary['total_flows_captured']} "
		f"flows_saved={summary['total_flows_saved']} "
		f"packets_saved={summary['total_packets_saved']} "
		f"stop_reason={summary['stop_reason']}",
	)
	return summary


def main() -> None:
	args = parse_args()
	config = CaptureConfig(
		iface=args.iface,
		duration=args.duration,
		flush_interval_sec=args.flush_interval_sec,
		idle_timeout=args.idle_timeout,
		min_packets=args.min_packets,
		output_dir=args.output_dir,
		capture_label=args.capture_label,
		bpf_filter=args.bpf_filter,
		target_classes=tuple(x.strip() for x in args.target_classes.split(",") if x.strip()),
		keywords_json=args.keywords_json,
	)
	collect_traffic(config)


if __name__ == "__main__":
	main()
