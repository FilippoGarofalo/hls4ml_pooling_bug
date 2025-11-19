import glob
import os
import xml.etree.ElementTree as ET


def _coerce_value(raw):
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return raw
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _parse_result_file(path):
    tree = ET.parse(path)
    root = tree.getroot()
    meta = {
        'Args': root.attrib.get('bambu_args'),
        'Version': root.attrib.get('bambu_version'),
        'Timestamp': root.attrib.get('timestamp'),
        'Benchmark': root.attrib.get('benchmark_name'),
        'File': os.path.basename(path),
    }
    metrics = {}
    for node in root:
        metrics[node.tag] = _coerce_value(node.attrib.get('value'))
    return {'meta': meta, 'metrics': metrics}


def parse_bambu_report(hls_dir):
    """Parse bambu_results XML files from ``hls_dir``.

    Returns a dictionary with the parsed entries or ``None`` if no file exists.
    """

    pattern = os.path.join(hls_dir, 'bambu_results*.xml')
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None

    parsed = [_parse_result_file(path) for path in matches]
    return {'results': parsed, 'latest': parsed[-1]}
