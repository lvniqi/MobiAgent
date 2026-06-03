from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Iterator, Tuple
import json
import re
import traceback
from openai import OpenAI
import copy
import io, base64
from PIL import Image
import time

app = FastAPI()

decider_client = None
grounder_client = None
planner_client = None

use_qwen3 = False

PLANNER_PROMPT = '''
## 角色定义
你是一个用户意图识别和智能手机应用分类助手。你需要根据用户的任务描述，输出能完成用户指定任务的应用类别。

## 任务描述
用户想要完成的任务是："{task_description}"

## 应用分类列表

- 通讯：示例应用：微信；示例任务：帮我给xxx发一条微信

- 社交：示例应用：微博，知乎，小红书；示例任务：帮我关注博主xxx

- 外卖：示例应用：饿了么，美团；示例任务：帮我点一份麦当劳的汉堡外卖

- 网购：示例应用：淘宝；示例任务：帮我下单购买一条白色连衣裙

- 视频：示例应用：爱奇艺，bilibili；示例任务：帮我播放xxx的第一条视频

- 酒店：示例应用：携程，飞猪；示例任务：帮我订一间汉庭酒店的大床房

- 旅行：示例应用：12306；示例任务：帮我查询上海到北京，9月1日出发的火车票

- 音乐：示例应用：网易云音乐；示例任务：帮我放一首周杰伦的歌曲

- 地图：示例应用：高德地图；示例任务：帮我导航到上海人民广场

- 打车：示例应用：滴滴出行；示例任务：帮我打车到上海交通大学


## 输出格式
请严格按照以下JSON格式输出：
```json
{{
    "reasoning": "分析任务内容，说明你选择这个应用分类的原因",
    "class": "任务分类，只能从上述应用分类列表中选择一个"
}}
```

## 重要规则
1. 应用分类只能从上述列表中选择
2. 如果应用分类列表中，没有一类应用能够贴合用户需求，"class"字段中请返回空字符串，也就是""
3. 类别必须完全匹配列表中的类别名，不能进行任何修改
4. 请你综合考虑每类应用的应用场景、示例任务，以及用户的实际输入，进行选择
'''.strip()

rewrite_cache = {}
task_refcnt = {}

terminate_checklist = [
    "当前页面未按预期加载",
    "进入了错误的页面",
    "打开了不合预期的页面",
    "当前打开了错误页面",
    "当前页面不合预期",
    "需要用户介入",
    "需要用户接管",
]

class_default_app = {
    "通讯": "微信",
    "社交": "小红书",
    "外卖": "饿了么",
    "网购": "淘宝",
    "视频": "bilibili",
    "酒店": "携程",
    "旅行": "携程",
    "音乐": "网易云",
    "地图": "高德",
    "打车": "高德",
}

supported_apps = {
    "支付宝": "com.eg.android.AlipayGphone",
    "微信": "com.tencent.mm",
    "QQ": "com.tencent.mobileqq",
    "微博": "com.sina.weibo",
    "新浪微博": "com.sina.weibo",
    "饿了么": "me.ele",
    "美团": "com.sankuai.meituan",
    "bilibili": "tv.danmaku.bili",
    "B站": "tv.danmaku.bili",
    "哔哩哔哩": "tv.danmaku.bili",
    "爱奇艺": "com.qiyi.video",
    "腾讯视频": "com.tencent.qqlive",
    "淘宝": "com.taobao.taobao",
    "京东": "com.jingdong.app.mall",
    "携程": "ctrip.android.view",
    "去哪儿": "com.Qunar",
    "知乎": "com.zhihu.android",
    "小红书": "com.xingin.xhs",
    "QQ音乐": "com.tencent.qqmusic",
    "网易云音乐": "com.netease.cloudmusic",
    "高德": "com.autonavi.minimap",
    "12306": "com.MobileTicket",
    "钉钉": "com.alibaba.android.rimet",
    "崩坏星穹铁道": "com.mihoyo.hyperion",
    "飞猪": "com.taobao.trip",
    "同程": "com.tongcheng.android",
    "华住会": "com.htinns",
    "酷狗音乐": "com.kugou.android",
    "汽水音乐": "com.luna.music",
    "百度地图": "com.baidu.BaiduMap",
    "百度贴吧": "com.baidu.tieba",
    "百度": "com.baidu.searchbox",
    "红果短剧": "com.phoenix.read",
    "闲鱼": "com.taobao.idlefish",
    "拼多多": "com.xunmeng.pinduoduo",
    "番茄小说": "com.dragon.read",
    "抖音": "com.ss.android.ugc.aweme",
    "QQ浏览器": "com.tencent.mtt",
    "今日头条": "com.ss.android.article.news",
    "快手": "com.smile.gifmaker",
    "喜马拉雅": "com.ximalaya.ting.android",
    "优酷": "com.youku.phone",
    "灵光": "com.antgroup.leopard.android",
    "千问": "com.aliyun.tongyi",
    "deepseek": "com.deepseek.chat",
    "豆包": "com.larus.nova",
    "夸克": "com.quark.browser",
    "云闪付": "com.unionpay",
    "大众点评": "com.dianping.v1",
    "作业帮": "com.baidu.homework",
    "剪映": "com.lemon.lv",
    "得物": "com.shizhuang.duapp",
    "番茄畅听": "com.xs.fm",
    "转转": "com.wuba.zhuanzhuan",
    "西瓜视频": "com.ss.android.article.video",
}

def should_terminate(reasoning: str):
    for phrase in terminate_checklist:
        if phrase in reasoning:
            return True
    return False

def try_find_app(task_description: str):
    longest_match = ""
    for app in supported_apps:
        if app.lower() in task_description.lower() and len(app) > len(longest_match):
            longest_match = app
    if longest_match != "":
        return longest_match, supported_apps[longest_match]
    else:
        return None, None

DECIDER_PROMPT = '''
You are a phone-use AI agent. Now your task is "{task}".
Your action history is:
{history}
Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action.
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT).
- Name: input, Parameters: text (the text to input).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: (no parameters).
Your output should be a JSON object with the following format:
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, wait, done)", "parameters": {{"param1": "value1", ...}}}}
Remember your task is "{task_repeat}".'''

GROUNDER_PROMPT = '''
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}'''

GROUNDER_PROMPT_QWEN3 = '''
Based on user's intent and the description of the target UI element, locate the element in the screenshot.
User's intent: {reasoning}
Target element's description: {description}
Report the bbox coordinates in JSON format.'''

class ResponseBody(BaseModel):
    reasoning: str
    action: str
    parameters: Dict[str, Any]

class RequestBody(BaseModel):
    task: str
    image: str
    history: List[str]

def sanitize_base64(image_b64: str) -> str:
    """Strip data-URI prefix and whitespace from base64 (Android Base64.DEFAULT adds newlines)."""
    if not image_b64:
        return image_b64
    if image_b64.startswith("data:") and "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return "".join(image_b64.split())

def parse_model_json(model_output: str):
    """Parse JSON from model output, tolerating markdown fences and extra text."""
    if model_output is None:
        raise ValueError("Empty model output")
    text = model_output.strip()

    def try_load(candidate: str):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    parsed = try_load(text)
    if parsed is not None:
        return parsed

    for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            parsed = try_load(match.group(1).strip())
            if parsed is not None:
                return parsed

    start_idx = text.find("{")
    if start_idx != -1:
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    parsed = try_load(text[start_idx:i + 1])
                    if parsed is not None:
                        return parsed
                    break

    start_idx = text.find("[")
    if start_idx != -1:
        bracket_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == "[":
                bracket_count += 1
            elif text[i] == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    parsed = try_load(text[start_idx:i + 1])
                    if parsed is not None:
                        return parsed
                    break

    raise ValueError(f"Cannot parse JSON from model output: {text[:300]}")

def extract_bbox(grounder_output_json, width: int, height: int, use_qwen3: bool):
    bbox = None

    if isinstance(grounder_output_json, list):
        if len(grounder_output_json) >= 4 and all(
            isinstance(v, (int, float)) for v in grounder_output_json[:4]
        ):
            bbox = grounder_output_json[:4]
        elif grounder_output_json and isinstance(grounder_output_json[0], dict):
            grounder_output_json = grounder_output_json[0]
        else:
            raise ValueError(f"Unexpected list format in grounder response: {grounder_output_json}")

    if bbox is None:
        if not isinstance(grounder_output_json, dict):
            raise ValueError(f"Unexpected grounder response type: {type(grounder_output_json)}")
        for key, value in grounder_output_json.items():
            if key.lower() in ["bbox", "bbox_2d", "bbox-2d", "bbox2d"]:
                bbox = value
                break
    if bbox is None:
        raise ValueError(f"No bbox field in grounder response: {grounder_output_json}")

    bbox = list(bbox)
    if use_qwen3:
        bbox[0] = bbox[0] / 1000 * width
        bbox[2] = bbox[2] / 1000 * width
        bbox[1] = bbox[1] / 1000 * height
        bbox[3] = bbox[3] / 1000 * height
    return bbox

def format_sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def build_messages(prompt: str, image_b64: str | None = None):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    if image_b64 is not None:
        image_b64 = sanitize_base64(image_b64)
        messages[0]["content"].insert(
            0,
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        )
    return messages

def iter_model_stream(model_client, prompt, image_b64=None) -> Iterator[str]:
    messages = build_messages(prompt, image_b64)
    stream = model_client.chat.completions.create(
        model="",
        messages=messages,
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta

def get_model_output(model_client, prompt, image_b64=None):
    start = time.perf_counter()
    parts = list(iter_model_stream(model_client, prompt, image_b64))
    content = "".join(parts)
    print(f"Model response time: {time.perf_counter() - start:.2f} seconds")
    return content

def rewrite_task(original_task: str, app_name):
    ret = original_task
    # add your custom rewriting rules here
    return ret

def validate_history(history: List[str]):
    filtered = []
    allowed_keys = {
        "click": {"target_element"},
        "input": {"text"},
        "swipe": {"direction"},
        "done": {}
    }
    for h in history:
        old = json.loads(h)
        new = copy.deepcopy(old)
        action = old["action"]
        if action not in allowed_keys:
            continue
        for k in old["parameters"]:
            if k not in allowed_keys[action]:
                new["parameters"].pop(k)
        filtered.append(new)
    
    return [json.dumps(act, ensure_ascii=False) for act in filtered]

# async 
def cleanup_task(task):
    # async with cache_lock:
    if task_refcnt.get(task, 0) > 0:
        task_refcnt[task] -= 1
        if task_refcnt[task] == 0:
            rewrite_cache.pop(task, None)
            task_refcnt.pop(task, None)

def process_v1_events(request_body: RequestBody) -> Iterator[Tuple[str, dict]]:
    if request_body.task.strip() == "":
        yield ("result", {
            "reasoning": "任务不能为空，任务终止",
            "action": "terminate",
            "parameters": {},
        })
        return

    history = request_body.history
    task = request_body.task

    if len(history) == 0:
        try:
            print(f"Task: {task}")
            yield ("progress", {"stage": "planner", "message": "正在识别目标应用..."})
            app_name, package_name = try_find_app(task)
            if app_name is None:
                planner_prompt = PLANNER_PROMPT.format(task_description=task)
                planner_parts = []
                for delta in iter_model_stream(planner_client, planner_prompt):
                    planner_parts.append(delta)
                    yield ("progress", {"stage": "planner", "delta": delta})
                planner_output = "".join(planner_parts)
                print(planner_output)
                planner_output_json = parse_model_json(planner_output)
                classification = planner_output_json["class"]
                if classification not in class_default_app:
                    app_name, package_name = None, ""
                else:
                    app_name = class_default_app[classification]
                    package_name = supported_apps[app_name]
        except Exception:
            traceback.print_exc()
            app_name, package_name = None, ""

        if app_name is None or app_name == "" or package_name == "":
            yield ("result", {
                "reasoning": f"暂不支持用户任务\"{task}\"需要打开的应用，任务终止",
                "action": "terminate",
                "parameters": {},
            })
            return

        reasoning = f"为了完成用户任务\"{task}\", 我需要打开应用\"{app_name}\""
        task_refcnt[task] = task_refcnt.get(task, 0) + 1
        if task_refcnt[task] == 1:
            rewrite_cache[task] = rewrite_task(task, app_name)
        yield ("result", {
            "reasoning": reasoning,
            "action": "open_app",
            "parameters": {"package_name": package_name},
        })
        return

    rewritten_task = rewrite_cache.get(task, task)
    history = validate_history(history)
    history_str = "(No history)" if len(history) == 0 else "\n".join(
        f"{idx}. {act}" for idx, act in enumerate(history, start=1)
    )

    img_b64 = sanitize_base64(request_body.image)
    pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
    width, height = pil_img.size
    print(f"Received image of size: {width}x{height}")

    decider_prompt = DECIDER_PROMPT.format(
        task=rewritten_task,
        history=history_str,
        task_repeat=rewritten_task,
    )
    yield ("progress", {"stage": "decider", "message": "正在分析截图..."})
    decider_parts = []
    for delta in iter_model_stream(decider_client, decider_prompt, img_b64):
        decider_parts.append(delta)
        yield ("progress", {"stage": "decider", "delta": delta})
    decider_output = "".join(decider_parts)
    print(decider_output)
    decider_output_json = parse_model_json(decider_output)
    reasoning = decider_output_json["reasoning"]
    if should_terminate(reasoning):
        cleanup_task(task)
        yield ("result", {
            "reasoning": reasoning,
            "action": "terminate",
            "parameters": {},
        })
        return

    action = decider_output_json["action"]
    parameters = decider_output_json["parameters"]
    if action == "click":
        grounder_prompt_fmt = GROUNDER_PROMPT_QWEN3 if use_qwen3 else GROUNDER_PROMPT
        grounder_prompt = grounder_prompt_fmt.format(
            reasoning=reasoning,
            description=parameters["target_element"],
        )
        yield ("progress", {"stage": "grounder", "message": "正在定位元素..."})
        grounder_parts = []
        for delta in iter_model_stream(grounder_client, grounder_prompt, img_b64):
            grounder_parts.append(delta)
            yield ("progress", {"stage": "grounder", "delta": delta})
        grounder_output = "".join(grounder_parts)
        print(grounder_output)
        grounder_output_json = parse_model_json(grounder_output)
        bbox = extract_bbox(grounder_output_json, width, height, use_qwen3)
        parameters["x"] = int((bbox[0] + bbox[2]) // 2)
        parameters["y"] = int((bbox[1] + bbox[3]) // 2)
    elif action == "done":
        cleanup_task(task)

    yield ("result", {
        "reasoning": reasoning,
        "action": action,
        "parameters": parameters,
    })

def build_v1_response(request_body: RequestBody) -> ResponseBody:
    for event, data in process_v1_events(request_body):
        if event == "result":
            return ResponseBody(**data)
    raise HTTPException(status_code=500, detail="No result produced")

def sse_response(request_body: RequestBody) -> Iterator[str]:
    try:
        for event, data in process_v1_events(request_body):
            yield format_sse(event, data)
    except Exception as e:
        traceback.print_exc()
        cleanup_task(request_body.task)
        yield format_sse("error", {"detail": str(e)})

@app.post("/v1")
async def v1(request_body: RequestBody):
    return StreamingResponse(
        sse_response(request_body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/v1/sync", response_model=ResponseBody)
async def v1_sync(request_body: RequestBody):
    try:
        return build_v1_response(request_body)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        cleanup_task(request_body.task)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Optional: Add a root endpoint for health checks
@app.get("/")
async def root():
    return {"message": "Welcome to the Simple FastAPI Server! Use /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=22334)
    parser.add_argument("--planner_url", type=str, help="Base URL for planner model service")
    parser.add_argument("--decider_url", type=str, help="Base URL for decider model service")
    parser.add_argument("--grounder_url", type=str, help="Base URL for grounder model service")
    parser.add_argument("--api_key", type=str, default="0", help="API key for model services")
    parser.add_argument("--use_qwen3", action='store_true', help="Use Qwen3-VL model format")
    args = parser.parse_args()
    use_qwen3 = args.use_qwen3
    decider_client = OpenAI(api_key=args.api_key, base_url=args.decider_url)
    grounder_client = OpenAI(api_key=args.api_key, base_url=args.grounder_url)
    planner_client = OpenAI(api_key=args.api_key, base_url=args.planner_url)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
