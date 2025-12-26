# coding: utf-8
import os
import time
import json
import random
import requests
import concurrent.futures
from tqdm import tqdm  # 进度条库
import base64

# --- 高级配置 ---
TOTAL_COUNT = 1100       # 目标采集数量(成功保存的组数)
MAX_WORKERS = 1          # 线程数 (没有代理IP建议设为 1，有代理可设为 5-10)

# 始终以脚本所在目录为基准，避免从不同工作目录运行导致路径跑偏
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
SAVE_DIR = os.path.join(_PROJECT_ROOT, 'data', 'raw')

APP_ID = '198420051'

# 代理配置 (如果你买了代理IP，填在这里)
# 格式示例: "http://user:pass@ip:port"
# 如果留空，则使用本机 IP (请务必把 MAX_WORKERS 设为 1 并增加延时)
PROXY_URL = ""

# 网络与鲁棒性配置
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3                 # 单次任务重试次数(网络/偶发失败)
BACKOFF_BASE = 0.8              # 退避基数(秒)
PRINT_FAIL_REASON = False       # True 时会打印失败原因(会更吵，但便于调试)

# 随机 UA 池 (模拟不同设备，降低特征)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-S908B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
]
# ----------------

os.makedirs(SAVE_DIR, exist_ok=True)

def get_proxy():
    if PROXY_URL:
        return {"http": PROXY_URL, "https": PROXY_URL}
    return None

def _sleep_jitter():
    # 随机延时 (如果是单线程，延时久一点；多线程则短一点)
    if MAX_WORKERS == 1:
        time.sleep(random.uniform(1.0, 3.0))
    else:
        time.sleep(random.uniform(0.1, 0.6))

def _make_session():
    s = requests.Session()
    proxies = get_proxy()
    if proxies:
        s.proxies.update(proxies)
    # 适度复用连接即可；更复杂的连接池配置不强依赖
    return s

def _build_headers():
    ua_str = random.choice(USER_AGENTS)
    headers = {
        'Host': 'turing.captcha.qcloud.com',
        'Connection': 'keep-alive',
        'User-Agent': ua_str,
        'Accept': '*/*',
        'Referer': 'https://turing.captcha.qcloud.com/',
        # 下面这些字段可按需打开/调整，用于更贴近真实请求
        # 'X-Requested-With': 'com.qidian.QDReader',
        # 'Accept-Language': 'zh-CN,zh-TW;q=0.9,zh;q=0.8,en-US;q=0.7,en;q=0.6',
    }
    # 与 tools/Captcha.py 一致：ua 参数为 User-Agent 的 base64
    ua_param = base64.b64encode(ua_str.encode('utf-8')).decode('utf-8')
    return headers, ua_param

def download_one_set(_task_id: int):
    """
    下载单组数据的完整流程
    返回: (ok: bool, reason: str)
    """
    session = _make_session()

    for attempt in range(1, MAX_RETRIES + 1):
        headers, ua_param = _build_headers()

        try:
            _sleep_jitter()

            # 1) 请求配置
            url = 'https://turing.captcha.qcloud.com/cap_union_prehandle'
            params = {
                'aid': APP_ID,
                'protocol': 'https',
                'accver': '1',
                'showtype': 'embed',
                # 使用真实生成的 ua（base64(User-Agent)）替换占位符
                'ua': ua_param,
                'noheader': '1',
                'fb': '1',
                'filter': '1',
            }

            resp = session.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                raise RuntimeError(f"prehandle_status_{resp.status_code}")

            content = resp.text or ""
            if '(' not in content or ')' not in content:
                return False, "non_jsonp_or_blocked"

            json_str = content[content.find('(') + 1: content.rfind(')')]
            try:
                config = json.loads(json_str)
            except json.JSONDecodeError:
                return False, "json_decode_error"

            # 仅筛点选类型
            if 'sess' not in config:
                return False, "no_sess"

            dyn_info = config.get('data', {}).get('dyn_show_info')
            if not dyn_info:
                return False, "no_dyn_show_info"

            bg_url = dyn_info.get('bg_elem_cfg', {}).get('img_url')
            icon_url = dyn_info.get('sprite_url')

            if not bg_url or not icon_url:
                return False, "no_image_url"

            if not bg_url.startswith('http'):
                bg_url = 'https://turing.captcha.qcloud.com' + bg_url
            if not icon_url.startswith('http'):
                icon_url = 'https://turing.captcha.qcloud.com' + icon_url

            # 2) 下载图片(流式)
            bg_resp = session.get(bg_url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True)
            icon_resp = session.get(icon_url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True)

            if bg_resp.status_code != 200 or icon_resp.status_code != 200:
                return False, f"img_status_{bg_resp.status_code}_{icon_resp.status_code}"

            # 3) 保存
            timestamp = int(time.time() * 1000)
            prefix = f"{timestamp}_{random.randint(1000,9999)}"

            bg_path = os.path.join(SAVE_DIR, f"{prefix}_bg.jpg")
            icon_path = os.path.join(SAVE_DIR, f"{prefix}_icon.png")

            with open(bg_path, 'wb') as f:
                for chunk in bg_resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)

            with open(icon_path, 'wb') as f:
                for chunk in icon_resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)

            return True, "ok"

        except Exception as e:
            # 退避重试(仅对异常/网络类问题)
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.3))
                continue
            return False, f"exception:{type(e).__name__}"

    return False, "unknown"


def main():
    print(
        f"开始批量采集 | 目标(成功): {TOTAL_COUNT} | 线程: {MAX_WORKERS} | 代理: {'开启' if PROXY_URL else '关闭'}\n"
        f"保存目录: {os.path.abspath(SAVE_DIR)}"
    )

    success_count = 0
    fail_stats = {}

    # 按“成功数达到目标”推进，而不是固定尝试 TOTAL_COUNT 次
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pending = set()

        def submit_one():
            # task id 仅用于区分，无业务含义
            return executor.submit(download_one_set, random.randint(1, 10**9))

        # 先填满并发
        for _ in range(max(1, MAX_WORKERS)):
            pending.add(submit_one())

        with tqdm(total=TOTAL_COUNT, unit="set") as pbar:
            while success_count < TOTAL_COUNT:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    ok, reason = fut.result()
                    if ok:
                        success_count += 1
                        pbar.update(1)
                    else:
                        fail_stats[reason] = fail_stats.get(reason, 0) + 1
                        if PRINT_FAIL_REASON:
                            print(f"[fail] {reason}")

                    # 补充一个新任务保持并发
                    if success_count < TOTAL_COUNT:
                        pending.add(submit_one())

    # 输出统计
    total_fail = sum(fail_stats.values())
    print(f"\n采集结束! 成功: {success_count} | 失败尝试: {total_fail}")
    if total_fail:
        top = sorted(fail_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        print("失败原因TOP10:")
        for k, v in top:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()