#!/usr/bin/env python3
"""Export the three MFDS DUR endpoints as Excel-readable CSV files."""

import argparse
import csv
import getpass
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree


BASE = "https://apis.data.go.kr/1471000/DURPrdlstInfoService03"
ENDPOINTS = {
    "interaction": "getUsjntTabooInfoList03",
    "pregnancy": "getPwnmTabooInfoList03",
    "age": "getSpcifyAgrdeTabooInfoList03",
}
OTC_LABELS = {"일반", "일반의약품"}


class CollectionError(Exception):
    pass


def api_error(code, message, key):
    code = str(code or "").strip()
    detail = f"코드 {code}" if re.fullmatch(r"\d{1,5}", code) else "응답 코드 없음"
    message = unquote(str(message or ""))
    if key:
        message = message.replace(key, "[키 숨김]")
    message = " ".join(message.split())
    # URLs and query strings may include credentials, so omit them entirely.
    if message and not re.search(r"https?://|servicekey\s*=", message, re.I):
        detail += f" · {message[:180]}"
    elif code == "11":
        detail += " · 필수 요청값을 확인하세요."
    return detail


def extract_items(body):
    items = body.get("items")
    if isinstance(items, dict):
        items = items.get("item", items)
    if items is None or items == "" or items == {}:
        return []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise CollectionError("응답의 items 형식을 확인할 수 없습니다.")
    return items


def decode_response(payload, key=""):
    try:
        outer = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # The portal may report errors as XML even when JSON was requested.
        try:
            error = ElementTree.fromstring(payload)
            code = error.findtext(".//returnReasonCode") or error.findtext(".//resultCode")
            message = error.findtext(".//returnAuthMsg") or error.findtext(".//resultMsg")
        except ElementTree.ParseError:
            code = message = None
        raise CollectionError(f"JSON 응답을 받지 못했습니다. {api_error(code, message, key)}") from None
    if not isinstance(outer, dict):
        raise CollectionError("응답의 최상위 객체를 확인할 수 없습니다.")
    response = outer.get("response", outer)
    if not isinstance(response, dict):
        raise CollectionError("응답의 response 객체를 확인할 수 없습니다.")
    header = response.get("header") or {}
    code = str(header.get("resultCode", "")).strip() if isinstance(header, dict) else ""
    body = response.get("body")
    if code not in {"0", "00", "000", "0000"} or not isinstance(body, dict):
        message = header.get("resultMsg") if isinstance(header, dict) else ""
        raise CollectionError(f"DUR API 오류. {api_error(code, message, key)}")
    try:
        total_count = int(body["totalCount"])
        page_no = int(body["pageNo"])
    except (KeyError, ValueError, TypeError):
        raise CollectionError("응답의 totalCount 또는 pageNo를 읽을 수 없습니다.") from None
    if total_count < 0 or page_no < 1:
        raise CollectionError("응답에 유효하지 않은 페이지 번호 또는 전체 건수가 있습니다.")
    return total_count, page_no, extract_items(body)


def fetch_page(endpoint, key, page_no, page_size, type_name=None):
    params = {"serviceKey": key, "pageNo": page_no, "numOfRows": page_size, "type": "json"}
    if type_name:
        params["typeName"] = type_name
    query = urlencode(params)
    request = Request(f"{BASE}/{endpoint}?{query}", headers={"Accept": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                payload = response.read()
        except HTTPError as error:
            status = error.code
            if status not in {429, 500, 502, 503, 504} or attempt == 2:
                try:
                    decode_response(error.read(65_536), key)
                except CollectionError as detail:
                    raise CollectionError(f"HTTP {status}. {detail}") from None
                raise CollectionError(f"HTTP {status}.") from None
        except (URLError, TimeoutError):
            if attempt == 2:
                raise CollectionError("DUR API에 연결하지 못했습니다. 서버의 네트워크 상태를 확인하세요.") from None
        else:
            return decode_response(payload, key)
        time.sleep(2 ** attempt)


def is_otc(value):
    return "".join(str(value or "").split()) in OTC_LABELS


def include_row(kind, row):
    if is_otc(row.get("ETC_OTC_NAME")):
        return True
    return kind == "interaction" and is_otc(row.get("MIXTURE_ETC_OTC_NAME"))


def write_excel_csv(source, destination, fields):
    with source.open(encoding="utf-8") as records, destination.open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for line in records:
            row = json.loads(line)
            writer.writerow({name: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for name, value in row.items()})


def collect(kind, endpoint, key, output, page_size, max_pages, include_all, type_name=None):
    scanned = stored = page_no = 0
    first_total = None
    fields = dict.fromkeys(("DUR_SEQ", "TYPE_NAME", "ITEM_SEQ", "ITEM_NAME", "ETC_OTC_NAME"))
    spool = output / f"{kind}.jsonl"
    with spool.open("w", encoding="utf-8") as rows:
        while True:
            page_no += 1
            try:
                total, returned_page, items = fetch_page(endpoint, key, page_no, page_size, type_name)
            except CollectionError as error:
                raise CollectionError(f"{endpoint} (페이지 {page_no}, 요청 행 수 {page_size}): {error}") from None
            if returned_page != page_no:
                raise CollectionError(f"{kind} 페이지 {page_no} 요청에 다른 페이지를 받았습니다.")
            if first_total is None:
                first_total = total
            if total != first_total:
                raise CollectionError(f"{kind} 조회 중 원본 건수가 바뀌었습니다. 다시 수집해 주세요.")
            if (not items and scanned < total) or scanned + len(items) > total:
                raise CollectionError(f"{kind} 페이지 {page_no}의 실제 행 수가 전체 건수와 일치하지 않습니다.")
            scanned += len(items)
            for item in items:
                fields.update(dict.fromkeys(item))
                if include_all or include_row(kind, item):
                    rows.write(json.dumps(item, ensure_ascii=False) + "\n")
                    stored += 1
            if page_no == 1 or page_no % 50 == 0 or scanned == total:
                print(f"{kind}  조회 {scanned:,}/{total:,}건 · 저장 대상 {stored:,}건", flush=True)
            if scanned == total or (max_pages is not None and page_no >= max_pages):
                break
    write_excel_csv(spool, output / f"dur_{kind}.csv", list(fields))
    spool.unlink()
    return {"endpoint": endpoint, "pages": page_no, "total": total, "scanned": scanned, "saved": stored,
            "complete": scanned == total, "otc_only": not include_all}


def read_service_key():
    key = os.environ.get("DUR_SERVICE_KEY", "").strip()
    if key:
        return key
    config_path = Path(__file__).resolve().with_name(".env")
    try:
        lines = config_path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeError):
        raise CollectionError("collect_dur.py 옆의 .env 파일을 UTF-8로 저장했는지 확인하세요.") from None
    for line in lines:
        name, separator, value = line.partition("=")
        if separator and name.strip() == "DUR_SERVICE_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] in {'"', "'"} and value[-1] == value[0]:
                value = value[1:-1].strip()
            return value
    return ""


def main():
    parser = argparse.ArgumentParser(description="식약처 DUR 병용·임부·연령금기 CSV 다운로드")
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("dur_csv"))
    parser.add_argument("--page-size", type=int, default=500, help="페이지당 요청 건수 (기본값 500)")
    parser.add_argument("--max-pages", type=int, help="연결 확인 시 유형별 최대 페이지 수")
    parser.add_argument("--include-all", action="store_true", help="일반약 선별 없이 응답 전체를 저장")
    parser.add_argument("--kind", choices=ENDPOINTS, help="지정한 금기 유형 하나만 조회")
    parser.add_argument("--type-name", help="포털 테스트와 동일하게 맞출 DUR 유형 값. --kind와 함께 사용")
    args = parser.parse_args()
    if args.page_size < 1 or (args.max_pages is not None and args.max_pages < 1):
        parser.error("페이지당 건수와 페이지 수는 양수여야 합니다.")
    if args.type_name and not args.kind:
        parser.error("--type-name을 지정하려면 --kind도 입력하세요.")
    target = args.output_dir.resolve()
    if target.exists():
        parser.error("새 출력 폴더를 지정하세요.")
    try:
        key = read_service_key()
    except CollectionError as error:
        parser.exit(1, f"수집 실패: {error}\n")
    if not key and sys.stdin.isatty():
        key = getpass.getpass("공공데이터포털 DUR 서비스 키: ").strip()
    if not key:
        parser.error(".env, DUR_SERVICE_KEY 환경 변수 또는 터미널의 숨김 입력으로 키를 제공하세요.")
    key = unquote(key)

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".dur-download-", dir=target.parent) as stage:
            ready = Path(stage) / "csv"
            ready.mkdir()
            summary = {}
            for kind in (args.kind,) if args.kind else ENDPOINTS:
                summary[kind] = collect(kind, ENDPOINTS[kind], key, ready, args.page_size,
                                        args.max_pages, args.include_all, args.type_name)
            (ready / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            ready.rename(target)
    except (CollectionError, OSError) as error:
        parser.exit(1, f"수집 실패: {error}\n")
    print(f"CSV 저장 완료: {target}")


if __name__ == "__main__":
    main()
