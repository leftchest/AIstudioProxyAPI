"""
PageController模块
封装了所有与Playwright页面直接交互的复杂逻辑。
"""
import asyncio
from typing import Callable, List, Dict, Any, Optional

from playwright.async_api import Page as AsyncPage, expect as expect_async, TimeoutError

from config import (
    TEMPERATURE_INPUT_SELECTOR, MAX_OUTPUT_TOKENS_SELECTOR, STOP_SEQUENCE_INPUT_SELECTOR,
    MAT_CHIP_REMOVE_BUTTON_SELECTOR, TOP_P_INPUT_SELECTOR, SUBMIT_BUTTON_SELECTOR,
    CLEAR_CHAT_BUTTON_SELECTOR, CLEAR_CHAT_CONFIRM_BUTTON_SELECTOR, OVERLAY_SELECTOR,
    PROMPT_TEXTAREA_SELECTOR, RESPONSE_CONTAINER_SELECTOR, RESPONSE_TEXT_SELECTOR,
    EDIT_MESSAGE_BUTTON_SELECTOR,USE_URL_CONTEXT_SELECTOR,UPLOAD_BUTTON_SELECTOR,
    SET_THINKING_BUDGET_TOGGLE_SELECTOR, THINKING_BUDGET_INPUT_SELECTOR,
    GROUNDING_WITH_GOOGLE_SEARCH_TOGGLE_SELECTOR
)
from config import (
    CLICK_TIMEOUT_MS, WAIT_FOR_ELEMENT_TIMEOUT_MS, CLEAR_CHAT_VERIFY_TIMEOUT_MS,
    DEFAULT_TEMPERATURE, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_STOP_SEQUENCES, DEFAULT_TOP_P,
    ENABLE_URL_CONTEXT, ENABLE_THINKING_BUDGET, DEFAULT_THINKING_BUDGET, ENABLE_GOOGLE_SEARCH
)
from models import ClientDisconnectedError
from .operations import save_error_snapshot, _wait_for_response_completion, _get_final_response_content

class PageController:
    """封装了与AI Studio页面交互的所有操作。"""

    def __init__(self, page: AsyncPage, logger, req_id: str):
        self.page = page
        self.logger = logger
        self.req_id = req_id

    async def _check_disconnect(self, check_client_disconnected: Callable, stage: str):
        """检查客户端是否断开连接。"""
        if check_client_disconnected(stage):
            raise ClientDisconnectedError(f"[{self.req_id}] Client disconnected at stage: {stage}")

    async def adjust_parameters(self, request_params: Dict[str, Any], page_params_cache: Dict[str, Any], params_cache_lock: asyncio.Lock, model_id_to_use: str, parsed_model_list: List[Dict[str, Any]], check_client_disconnected: Callable):
        """调整所有请求参数。"""
        self.logger.info(f"[{self.req_id}] 开始调整所有请求参数...")
        await self._check_disconnect(check_client_disconnected, "Start Parameter Adjustment")

        # 调整温度
        temp_to_set = request_params.get('temperature', DEFAULT_TEMPERATURE)
        await self._adjust_temperature(temp_to_set, page_params_cache, params_cache_lock, check_client_disconnected)
        await self._check_disconnect(check_client_disconnected, "After Temperature Adjustment")

        # 调整最大Token
        max_tokens_to_set = request_params.get('max_output_tokens', DEFAULT_MAX_OUTPUT_TOKENS)
        await self._adjust_max_tokens(max_tokens_to_set, page_params_cache, params_cache_lock, model_id_to_use, parsed_model_list, check_client_disconnected)
        await self._check_disconnect(check_client_disconnected, "After Max Tokens Adjustment")

        # 调整停止序列
        stop_to_set = request_params.get('stop', DEFAULT_STOP_SEQUENCES)
        await self._adjust_stop_sequences(stop_to_set, page_params_cache, params_cache_lock, check_client_disconnected)
        await self._check_disconnect(check_client_disconnected, "After Stop Sequences Adjustment")

        # 调整Top P
        top_p_to_set = request_params.get('top_p', DEFAULT_TOP_P)
        await self._adjust_top_p(top_p_to_set, check_client_disconnected)
        await self._check_disconnect(check_client_disconnected, "End Parameter Adjustment")

        # 调整URL CONTEXT
        if ENABLE_URL_CONTEXT:
            await self._open_url_content(check_client_disconnected)
        else:
            self.logger.info(f"[{self.req_id}] URL Context 功能已禁用，跳过调整。")

        # 调整“思考预算”
        await self._handle_thinking_budget(request_params, check_client_disconnected)

        # 调整 Google Search 开关
        await self._adjust_google_search(request_params, check_client_disconnected)

    async def _handle_thinking_budget(self, request_params: Dict[str, Any], check_client_disconnected: Callable):
        """处理思考预算的调整逻辑。"""
        reasoning_effort = request_params.get('reasoning_effort')
        if reasoning_effort is not None:
            # 用户指定了，则开启并设置
            self.logger.info(f"[{self.req_id}] 用户指定了 reasoning_effort: {reasoning_effort}。")
            await self._control_thinking_budget_toggle(should_be_checked=True, check_client_disconnected=check_client_disconnected)
            await self._adjust_thinking_budget(reasoning_effort, check_client_disconnected)
        else:
            # 用户未指定，根据默认配置
            self.logger.info(f"[{self.req_id}] 用户未指定 reasoning_effort，根据默认配置 ENABLE_THINKING_BUDGET: {ENABLE_THINKING_BUDGET}。")
            await self._control_thinking_budget_toggle(should_be_checked=ENABLE_THINKING_BUDGET, check_client_disconnected=check_client_disconnected)
            if ENABLE_THINKING_BUDGET:
                # 如果默认开启，则使用默认值
                await self._adjust_thinking_budget(None, check_client_disconnected)

    def _parse_thinking_budget(self, reasoning_effort: Optional[Any]) -> Optional[int]:
        """从 reasoning_effort 解析出 token_budget。"""
        token_budget = None
        if reasoning_effort is None:
            token_budget = DEFAULT_THINKING_BUDGET
            self.logger.info(f"[{self.req_id}] 'reasoning_effort' 为空，使用默认思考预算: {token_budget}")
        elif isinstance(reasoning_effort, int):
            token_budget = reasoning_effort
        elif isinstance(reasoning_effort, str):
            if reasoning_effort.lower() == 'none':
                token_budget = DEFAULT_THINKING_BUDGET
                self.logger.info(f"[{self.req_id}] 'reasoning_effort' 为 'none' 字符串，使用默认思考预算: {token_budget}")
            else:
                effort_map = {
                    "low": 1000,
                    "medium": 8000,
                    "high": 24000
                }
                token_budget = effort_map.get(reasoning_effort.lower())
                if token_budget is None:
                    try:
                        token_budget = int(reasoning_effort)
                    except (ValueError, TypeError):
                        pass # token_budget remains None
        
        if token_budget is None:
             self.logger.warning(f"[{self.req_id}] 无法从 '{reasoning_effort}' (类型: {type(reasoning_effort)}) 解析出有效的 token_budget。")

        return token_budget

    async def _adjust_thinking_budget(self, reasoning_effort: Optional[Any], check_client_disconnected: Callable):
        """根据 reasoning_effort 调整思考预算。"""
        self.logger.info(f"[{self.req_id}] 检查并调整思考预算，输入值: {reasoning_effort}")
        
        token_budget = self._parse_thinking_budget(reasoning_effort)

        if token_budget is None:
            self.logger.warning(f"[{self.req_id}] 无效的 reasoning_effort 值: '{reasoning_effort}'。跳过调整。")
            return

        budget_input_locator = self.page.locator(THINKING_BUDGET_INPUT_SELECTOR)
        
        try:
            await expect_async(budget_input_locator).to_be_visible(timeout=5000)
            await self._check_disconnect(check_client_disconnected, "思考预算调整 - 输入框可见后")
            
            self.logger.info(f"[{self.req_id}] 设置思考预算为: {token_budget}")
            await budget_input_locator.fill(str(token_budget), timeout=5000)
            await self._check_disconnect(check_client_disconnected, "思考预算调整 - 填充输入框后")

            # 验证
            await asyncio.sleep(0.1)
            new_value_str = await budget_input_locator.input_value(timeout=3000)
            if int(new_value_str) == token_budget:
                self.logger.info(f"[{self.req_id}] ✅ 思考预算已成功更新为: {new_value_str}")
            else:
                self.logger.warning(f"[{self.req_id}] ⚠️ 思考预算更新后验证失败。页面显示: {new_value_str}, 期望: {token_budget}")

        except Exception as e:
            self.logger.error(f"[{self.req_id}] ❌ 调整思考预算时出错: {e}")
            if isinstance(e, ClientDisconnectedError):
                raise

    def _should_enable_google_search(self, request_params: Dict[str, Any]) -> bool:
        """根据请求参数或默认配置决定是否应启用 Google Search。"""
        if 'tools' in request_params and request_params.get('tools') is not None:
            tools = request_params.get('tools')
            has_google_search_tool = False
            if isinstance(tools, list):
                for tool in tools:
                    if isinstance(tool, dict):
                        if tool.get('google_search_retrieval') is not None:
                            has_google_search_tool = True
                            break
                        if tool.get('function', {}).get('name') == 'googleSearch':
                            has_google_search_tool = True
                            break
            self.logger.info(f"[{self.req_id}] 请求中包含 'tools' 参数。检测到 Google Search 工具: {has_google_search_tool}。")
            return has_google_search_tool
        else:
            self.logger.info(f"[{self.req_id}] 请求中不包含 'tools' 参数。使用默认配置 ENABLE_GOOGLE_SEARCH: {ENABLE_GOOGLE_SEARCH}。")
            return ENABLE_GOOGLE_SEARCH

    async def _adjust_google_search(self, request_params: Dict[str, Any], check_client_disconnected: Callable):
        """根据请求参数或默认配置，双向控制 Google Search 开关。"""
        self.logger.info(f"[{self.req_id}] 检查并调整 Google Search 开关...")

        should_enable_search = self._should_enable_google_search(request_params)

        toggle_selector = GROUNDING_WITH_GOOGLE_SEARCH_TOGGLE_SELECTOR
        
        try:
            toggle_locator = self.page.locator(toggle_selector)
            await expect_async(toggle_locator).to_be_visible(timeout=5000)
            await self._check_disconnect(check_client_disconnected, "Google Search 开关 - 元素可见后")

            is_checked_str = await toggle_locator.get_attribute("aria-checked")
            is_currently_checked = is_checked_str == "true"
            self.logger.info(f"[{self.req_id}] Google Search 开关当前状态: '{is_checked_str}'。期望状态: {should_enable_search}")

            if should_enable_search != is_currently_checked:
                action = "打开" if should_enable_search else "关闭"
                self.logger.info(f"[{self.req_id}] Google Search 开关状态与期望不符。正在点击以{action}...")
                await toggle_locator.click(timeout=CLICK_TIMEOUT_MS)
                await self._check_disconnect(check_client_disconnected, f"Google Search 开关 - 点击{action}后")
                await asyncio.sleep(0.5) # 等待UI更新
                new_state = await toggle_locator.get_attribute("aria-checked")
                if (new_state == "true") == should_enable_search:
                    self.logger.info(f"[{self.req_id}] ✅ Google Search 开关已成功{action}。")
                else:
                    self.logger.warning(f"[{self.req_id}] ⚠️ Google Search 开关{action}失败。当前状态: '{new_state}'")
            else:
                self.logger.info(f"[{self.req_id}] Google Search 开关已处于期望状态，无需操作。")

        except Exception as e:
            self.logger.error(f"[{self.req_id}] ❌ 操作 'Google Search toggle' 开关时发生错误: {e}")
            if isinstance(e, ClientDisconnectedError):
                 raise

    async def _open_url_content(self,check_client_disconnected: Callable):
        try:
            collapse_tools_locator = self.page.locator('button[aria-label="Expand or collapse tools"]')
            grandparent_locator = collapse_tools_locator.locator("xpath=../..")

            # 3. 获取祖父级元素的 class 属性值
            # get_attribute 返回一个包含所有 class 的字符串，例如 "menu dropdown active"
            class_string = await grandparent_locator.get_attribute("class")

            # 4. 在 Python 中进行判断
            # 确保 class_string 不是 None，并且 'expanded' 是一个独立的 class
            if class_string and "expanded" not in class_string.split():
                await collapse_tools_locator.click(timeout=CLICK_TIMEOUT_MS)
                await asyncio.sleep(0.5)
            use_url_content_selector = self.page.locator(USE_URL_CONTEXT_SELECTOR)
            is_checked = await use_url_content_selector.get_attribute("aria-checked")
            if "false" == is_checked:
                await use_url_content_selector.click(timeout=CLICK_TIMEOUT_MS)
                await self._check_disconnect(check_client_disconnected, "点击URLCONTEXT")
        except Exception as e:
            self.logger.error(f"[{self.req_id}] ❌ 操作USE_URL_CONTEXT_SELECTOR时发生错误:{e}。")

    async def _control_thinking_budget_toggle(self, should_be_checked: bool, check_client_disconnected: Callable):
        """
        根据 should_be_checked 的值，控制 "Thinking Budget" 滑块开关的状态。
        """
        toggle_selector = SET_THINKING_BUDGET_TOGGLE_SELECTOR
        self.logger.info(f"[{self.req_id}] 控制 'Thinking Budget' 开关，期望状态: {'选中' if should_be_checked else '未选中'}...")

        try:
            toggle_locator = self.page.locator(toggle_selector)
            await expect_async(toggle_locator).to_be_visible(timeout=5000)
            await self._check_disconnect(check_client_disconnected, "思考预算开关 - 元素可见后")

            is_checked_str = await toggle_locator.get_attribute("aria-checked")
            current_state_is_checked = is_checked_str == "true"
            self.logger.info(f"[{self.req_id}] 思考预算开关当前 'aria-checked' 状态: {is_checked_str} (当前是否选中: {current_state_is_checked})")

            if current_state_is_checked != should_be_checked:
                action = "启用" if should_be_checked else "禁用"
                self.logger.info(f"[{self.req_id}] 思考预算开关当前状态与期望不符，正在点击以{action}...")
                await toggle_locator.click(timeout=CLICK_TIMEOUT_MS)
                await self._check_disconnect(check_client_disconnected, f"思考预算开关 - 点击{action}后")

                await asyncio.sleep(0.5)
                new_state_str = await toggle_locator.get_attribute("aria-checked")
                new_state_is_checked = new_state_str == "true"

                if new_state_is_checked == should_be_checked:
                    self.logger.info(f"[{self.req_id}] ✅ 'Thinking Budget' 开关已成功{action}。新状态: {new_state_str}")
                else:
                    self.logger.warning(f"[{self.req_id}] ⚠️ 'Thinking Budget' 开关{action}后验证失败。期望状态: '{should_be_checked}', 实际状态: '{new_state_str}'")
            else:
                self.logger.info(f"[{self.req_id}] 'Thinking Budget' 开关已处于期望状态，无需操作。")

        except Exception as e:
            self.logger.error(f"[{self.req_id}] ❌ 操作 'Thinking Budget toggle' 开关时发生错误: {e}")
            if isinstance(e, ClientDisconnectedError):
                raise
    async def _adjust_temperature(self, temperature: float, page_params_cache: dict, params_cache_lock: asyncio.Lock, check_client_disconnected: Callable):
        """调整温度参数。"""
        async with params_cache_lock:
            self.logger.info(f"[{self.req_id}] 检查并调整温度设置...")
            clamped_temp = max(0.0, min(2.0, temperature))
            if clamped_temp != temperature:
                self.logger.warning(f"[{self.req_id}] 请求的温度 {temperature} 超出范围 [0, 2]，已调整为 {clamped_temp}")

            cached_temp = page_params_cache.get("temperature")
            if cached_temp is not None and abs(cached_temp - clamped_temp) < 0.001:
                self.logger.info(f"[{self.req_id}] 温度 ({clamped_temp}) 与缓存值 ({cached_temp}) 一致。跳过页面交互。")
                return

            self.logger.info(f"[{self.req_id}] 请求温度 ({clamped_temp}) 与缓存值 ({cached_temp}) 不一致或缓存中无值。需要与页面交互。")
            temp_input_locator = self.page.locator(TEMPERATURE_INPUT_SELECTOR)


            try:
                await expect_async(temp_input_locator).to_be_visible(timeout=5000)
                await self._check_disconnect(check_client_disconnected, "温度调整 - 输入框可见后")

                current_temp_str = await temp_input_locator.input_value(timeout=3000)
                await self._check_disconnect(check_client_disconnected, "温度调整 - 读取输入框值后")

                current_temp_float = float(current_temp_str)
                self.logger.info(f"[{self.req_id}] 页面当前温度: {current_temp_float}, 请求调整后温度: {clamped_temp}")

                if abs(current_temp_float - clamped_temp) < 0.001:
                    self.logger.info(f"[{self.req_id}] 页面当前温度 ({current_temp_float}) 与请求温度 ({clamped_temp}) 一致。更新缓存并跳过写入。")
                    page_params_cache["temperature"] = current_temp_float
                else:
                    self.logger.info(f"[{self.req_id}] 页面温度 ({current_temp_float}) 与请求温度 ({clamped_temp}) 不同，正在更新...")
                    await temp_input_locator.fill(str(clamped_temp), timeout=5000)
                    await self._check_disconnect(check_client_disconnected, "温度调整 - 填充输入框后")

                    await asyncio.sleep(0.1)
                    new_temp_str = await temp_input_locator.input_value(timeout=3000)
                    new_temp_float = float(new_temp_str)

                    if abs(new_temp_float - clamped_temp) < 0.001:
                        self.logger.info(f"[{self.req_id}] ✅ 温度已成功更新为: {new_temp_float}。更新缓存。")
                        page_params_cache["temperature"] = new_temp_float
                    else:
                        self.logger.warning(f"[{self.req_id}] ⚠️ 温度更新后验证失败。页面显示: {new_temp_float}, 期望: {clamped_temp}。清除缓存中的温度。")
                        page_params_cache.pop("temperature", None)
                        await save_error_snapshot(f"temperature_verify_fail_{self.req_id}")

            except ValueError as ve:
                self.logger.error(f"[{self.req_id}] 转换温度值为浮点数时出错. 错误: {ve}。清除缓存中的温度。")
                page_params_cache.pop("temperature", None)
                await save_error_snapshot(f"temperature_value_error_{self.req_id}")
            except Exception as pw_err:
                self.logger.error(f"[{self.req_id}] ❌ 操作温度输入框时发生错误: {pw_err}。清除缓存中的温度。")
                page_params_cache.pop("temperature", None)
                await save_error_snapshot(f"temperature_playwright_error_{self.req_id}")
                if isinstance(pw_err, ClientDisconnectedError):
                    raise

    async def _adjust_max_tokens(self, max_tokens: int, page_params_cache: dict, params_cache_lock: asyncio.Lock, model_id_to_use: str, parsed_model_list: list, check_client_disconnected: Callable):
        """调整最大输出Token参数。"""
        async with params_cache_lock:
            self.logger.info(f"[{self.req_id}] 检查并调整最大输出 Token 设置...")
            min_val_for_tokens = 1
            max_val_for_tokens_from_model = 65536

            if model_id_to_use and parsed_model_list:
                current_model_data = next((m for m in parsed_model_list if m.get("id") == model_id_to_use), None)
                if current_model_data and current_model_data.get("supported_max_output_tokens") is not None:
                    try:
                        supported_tokens = int(current_model_data["supported_max_output_tokens"])
                        if supported_tokens > 0:
                            max_val_for_tokens_from_model = supported_tokens
                        else:
                            self.logger.warning(f"[{self.req_id}] 模型 {model_id_to_use} supported_max_output_tokens 无效: {supported_tokens}")
                    except (ValueError, TypeError):
                        self.logger.warning(f"[{self.req_id}] 模型 {model_id_to_use} supported_max_output_tokens 解析失败")

            clamped_max_tokens = max(min_val_for_tokens, min(max_val_for_tokens_from_model, max_tokens))
            if clamped_max_tokens != max_tokens:
                self.logger.warning(f"[{self.req_id}] 请求的最大输出 Tokens {max_tokens} 超出模型范围，已调整为 {clamped_max_tokens}")

            cached_max_tokens = page_params_cache.get("max_output_tokens")
            if cached_max_tokens is not None and cached_max_tokens == clamped_max_tokens:
                self.logger.info(f"[{self.req_id}] 最大输出 Tokens ({clamped_max_tokens}) 与缓存值一致。跳过页面交互。")
                return

            max_tokens_input_locator = self.page.locator(MAX_OUTPUT_TOKENS_SELECTOR)

            try:
                await expect_async(max_tokens_input_locator).to_be_visible(timeout=5000)
                await self._check_disconnect(check_client_disconnected, "最大输出Token调整 - 输入框可见后")

                current_max_tokens_str = await max_tokens_input_locator.input_value(timeout=3000)
                current_max_tokens_int = int(current_max_tokens_str)

                if current_max_tokens_int == clamped_max_tokens:
                    self.logger.info(f"[{self.req_id}] 页面当前最大输出 Tokens ({current_max_tokens_int}) 与请求值 ({clamped_max_tokens}) 一致。更新缓存并跳过写入。")
                    page_params_cache["max_output_tokens"] = current_max_tokens_int
                else:
                    self.logger.info(f"[{self.req_id}] 页面最大输出 Tokens ({current_max_tokens_int}) 与请求值 ({clamped_max_tokens}) 不同，正在更新...")
                    await max_tokens_input_locator.fill(str(clamped_max_tokens), timeout=5000)
                    await self._check_disconnect(check_client_disconnected, "最大输出Token调整 - 填充输入框后")

                    await asyncio.sleep(0.1)
                    new_max_tokens_str = await max_tokens_input_locator.input_value(timeout=3000)
                    new_max_tokens_int = int(new_max_tokens_str)

                    if new_max_tokens_int == clamped_max_tokens:
                        self.logger.info(f"[{self.req_id}] ✅ 最大输出 Tokens 已成功更新为: {new_max_tokens_int}")
                        page_params_cache["max_output_tokens"] = new_max_tokens_int
                    else:
                        self.logger.warning(f"[{self.req_id}] ⚠️ 最大输出 Tokens 更新后验证失败。页面显示: {new_max_tokens_int}, 期望: {clamped_max_tokens}。清除缓存。")
                        page_params_cache.pop("max_output_tokens", None)
                        await save_error_snapshot(f"max_tokens_verify_fail_{self.req_id}")

            except (ValueError, TypeError) as ve:
                self.logger.error(f"[{self.req_id}] 转换最大输出 Tokens 值时出错: {ve}。清除缓存。")
                page_params_cache.pop("max_output_tokens", None)
                await save_error_snapshot(f"max_tokens_value_error_{self.req_id}")
            except Exception as e:
                self.logger.error(f"[{self.req_id}] ❌ 调整最大输出 Tokens 时出错: {e}。清除缓存。")
                page_params_cache.pop("max_output_tokens", None)
                await save_error_snapshot(f"max_tokens_error_{self.req_id}")
                if isinstance(e, ClientDisconnectedError):
                    raise
    
    async def _adjust_stop_sequences(self, stop_sequences, page_params_cache: dict, params_cache_lock: asyncio.Lock, check_client_disconnected: Callable):
        """调整停止序列参数。"""
        async with params_cache_lock:
            self.logger.info(f"[{self.req_id}] 检查并设置停止序列...")

            # 处理不同类型的stop_sequences输入
            normalized_requested_stops = set()
            if stop_sequences is not None:
                if isinstance(stop_sequences, str):
                    # 单个字符串
                    if stop_sequences.strip():
                        normalized_requested_stops.add(stop_sequences.strip())
                elif isinstance(stop_sequences, list):
                    # 字符串列表
                    for s in stop_sequences:
                        if isinstance(s, str) and s.strip():
                            normalized_requested_stops.add(s.strip())

            cached_stops_set = page_params_cache.get("stop_sequences")

            if cached_stops_set is not None and cached_stops_set == normalized_requested_stops:
                self.logger.info(f"[{self.req_id}] 请求的停止序列与缓存值一致。跳过页面交互。")
                return

            stop_input_locator = self.page.locator(STOP_SEQUENCE_INPUT_SELECTOR)
            remove_chip_buttons_locator = self.page.locator(MAT_CHIP_REMOVE_BUTTON_SELECTOR)

            try:
                # 清空已有的停止序列
                initial_chip_count = await remove_chip_buttons_locator.count()
                removed_count = 0
                max_removals = initial_chip_count + 5

                while await remove_chip_buttons_locator.count() > 0 and removed_count < max_removals:
                    await self._check_disconnect(check_client_disconnected, "停止序列清除 - 循环开始")
                    try:
                        await remove_chip_buttons_locator.first.click(timeout=2000)
                        removed_count += 1
                        await asyncio.sleep(0.15)
                    except Exception:
                        break

                # 添加新的停止序列
                if normalized_requested_stops:
                    await expect_async(stop_input_locator).to_be_visible(timeout=5000)
                    for seq in normalized_requested_stops:
                        await stop_input_locator.fill(seq, timeout=3000)
                        await stop_input_locator.press("Enter", timeout=3000)
                        await asyncio.sleep(0.2)

                page_params_cache["stop_sequences"] = normalized_requested_stops
                self.logger.info(f"[{self.req_id}] ✅ 停止序列已成功设置。缓存已更新。")

            except Exception as e:
                self.logger.error(f"[{self.req_id}] ❌ 设置停止序列时出错: {e}")
                page_params_cache.pop("stop_sequences", None)
                await save_error_snapshot(f"stop_sequence_error_{self.req_id}")
                if isinstance(e, ClientDisconnectedError):
                    raise

    async def _adjust_top_p(self, top_p: float, check_client_disconnected: Callable):
        """调整Top P参数。"""
        self.logger.info(f"[{self.req_id}] 检查并调整 Top P 设置...")
        clamped_top_p = max(0.0, min(1.0, top_p))

        if abs(clamped_top_p - top_p) > 1e-9:
            self.logger.warning(f"[{self.req_id}] 请求的 Top P {top_p} 超出范围 [0, 1]，已调整为 {clamped_top_p}")

        top_p_input_locator = self.page.locator(TOP_P_INPUT_SELECTOR)
        try:
            await expect_async(top_p_input_locator).to_be_visible(timeout=5000)
            await self._check_disconnect(check_client_disconnected, "Top P 调整 - 输入框可见后")

            current_top_p_str = await top_p_input_locator.input_value(timeout=3000)
            current_top_p_float = float(current_top_p_str)

            if abs(current_top_p_float - clamped_top_p) > 1e-9:
                self.logger.info(f"[{self.req_id}] 页面 Top P ({current_top_p_float}) 与请求值 ({clamped_top_p}) 不同，正在更新...")
                await top_p_input_locator.fill(str(clamped_top_p), timeout=5000)
                await self._check_disconnect(check_client_disconnected, "Top P 调整 - 填充输入框后")

                # 验证设置是否成功
                await asyncio.sleep(0.1)
                new_top_p_str = await top_p_input_locator.input_value(timeout=3000)
                new_top_p_float = float(new_top_p_str)

                if abs(new_top_p_float - clamped_top_p) <= 1e-9:
                    self.logger.info(f"[{self.req_id}] ✅ Top P 已成功更新为: {new_top_p_float}")
                else:
                    self.logger.warning(f"[{self.req_id}] ⚠️ Top P 更新后验证失败。页面显示: {new_top_p_float}, 期望: {clamped_top_p}")
                    await save_error_snapshot(f"top_p_verify_fail_{self.req_id}")
            else:
                self.logger.info(f"[{self.req_id}] 页面 Top P ({current_top_p_float}) 与请求值 ({clamped_top_p}) 一致，无需更改")

        except (ValueError, TypeError) as ve:
            self.logger.error(f"[{self.req_id}] 转换 Top P 值时出错: {ve}")
            await save_error_snapshot(f"top_p_value_error_{self.req_id}")
        except Exception as e:
            self.logger.error(f"[{self.req_id}] ❌ 调整 Top P 时出错: {e}")
            await save_error_snapshot(f"top_p_error_{self.req_id}")
            if isinstance(e, ClientDisconnectedError):
                raise

    async def clear_chat_history(self, check_client_disconnected: Callable):
        """清空聊天记录。"""
        self.logger.info(f"[{self.req_id}] 开始清空聊天记录...")
        await self._check_disconnect(check_client_disconnected, "Start Clear Chat")

        try:
            # 一般是使用流式代理时遇到,流式输出已结束,但页面上AI仍回复个不停,此时会锁住清空按钮,但页面仍是/new_chat,而跳过后续清空操作
            # 导致后续请求无法发出而卡住,故先检查并点击发送按钮(此时是停止功能)
            submit_button_locator = self.page.locator(SUBMIT_BUTTON_SELECTOR)
            try:
                self.logger.info(f"[{self.req_id}] 尝试检查发送按钮状态...")
                # 使用较短的超时时间（1秒），避免长时间阻塞，因为这不是清空流程的常见步骤
                await expect_async(submit_button_locator).to_be_enabled(timeout=1000)
                self.logger.info(f"[{self.req_id}] 发送按钮可用，尝试点击并等待1秒...")
                await submit_button_locator.click(timeout=CLICK_TIMEOUT_MS)
                await asyncio.sleep(1.0)
                self.logger.info(f"[{self.req_id}] 发送按钮点击并等待完成。")
            except Exception as e_submit:
                # 如果发送按钮不可用、超时或发生Playwright相关错误，记录日志并继续
                self.logger.info(f"[{self.req_id}] 发送按钮不可用或检查/点击时发生Playwright错误。符合预期,继续检查清空按钮。")

            clear_chat_button_locator = self.page.locator(CLEAR_CHAT_BUTTON_SELECTOR)
            confirm_button_locator = self.page.locator(CLEAR_CHAT_CONFIRM_BUTTON_SELECTOR)
            overlay_locator = self.page.locator(OVERLAY_SELECTOR)

            can_attempt_clear = False
            try:
                await expect_async(clear_chat_button_locator).to_be_enabled(timeout=3000)
                can_attempt_clear = True
                self.logger.info(f"[{self.req_id}] \"清空聊天\"按钮可用，继续清空流程。")
            except Exception as e_enable:
                is_new_chat_url = '/prompts/new_chat' in self.page.url.rstrip('/')
                if is_new_chat_url:
                    self.logger.info(f"[{self.req_id}] \"清空聊天\"按钮不可用 (预期，因为在 new_chat 页面)。跳过清空操作。")
                else:
                    self.logger.warning(f"[{self.req_id}] 等待\"清空聊天\"按钮可用失败: {e_enable}。清空操作可能无法执行。")

            await self._check_disconnect(check_client_disconnected, "清空聊天 - \"清空聊天\"按钮可用性检查后")

            if can_attempt_clear:
                await self._execute_chat_clear(clear_chat_button_locator, confirm_button_locator, overlay_locator, check_client_disconnected)
                await self._verify_chat_cleared(check_client_disconnected)

        except Exception as e_clear:
            self.logger.error(f"[{self.req_id}] 清空聊天过程中发生错误: {e_clear}")
            if not (isinstance(e_clear, ClientDisconnectedError) or (hasattr(e_clear, 'name') and 'Disconnect' in e_clear.name)):
                await save_error_snapshot(f"clear_chat_error_{self.req_id}")
            raise

    async def _execute_chat_clear(self, clear_chat_button_locator, confirm_button_locator, overlay_locator, check_client_disconnected: Callable):
        """执行清空聊天操作"""
        overlay_initially_visible = False
        try:
            if await overlay_locator.is_visible(timeout=1000):
                overlay_initially_visible = True
                self.logger.info(f"[{self.req_id}] 清空聊天确认遮罩层已可见。直接点击\"继续\"。")
        except TimeoutError:
            self.logger.info(f"[{self.req_id}] 清空聊天确认遮罩层初始不可见 (检查超时或未找到)。")
            overlay_initially_visible = False
        except Exception as e_vis_check:
            self.logger.warning(f"[{self.req_id}] 检查遮罩层可见性时发生错误: {e_vis_check}。假定不可见。")
            overlay_initially_visible = False

        await self._check_disconnect(check_client_disconnected, "清空聊天 - 初始遮罩层检查后")

        if overlay_initially_visible:
            self.logger.info(f"[{self.req_id}] 点击\"继续\"按钮 (遮罩层已存在): {CLEAR_CHAT_CONFIRM_BUTTON_SELECTOR}")
            await confirm_button_locator.click(timeout=CLICK_TIMEOUT_MS)
        else:
            self.logger.info(f"[{self.req_id}] 点击\"清空聊天\"按钮: {CLEAR_CHAT_BUTTON_SELECTOR}")
            await clear_chat_button_locator.click(timeout=CLICK_TIMEOUT_MS)
            await self._check_disconnect(check_client_disconnected, "清空聊天 - 点击\"清空聊天\"后")

            try:
                self.logger.info(f"[{self.req_id}] 等待清空聊天确认遮罩层出现: {OVERLAY_SELECTOR}")
                await expect_async(overlay_locator).to_be_visible(timeout=WAIT_FOR_ELEMENT_TIMEOUT_MS)
                self.logger.info(f"[{self.req_id}] 清空聊天确认遮罩层已出现。")
            except TimeoutError:
                error_msg = f"等待清空聊天确认遮罩层超时 (点击清空按钮后)。请求 ID: {self.req_id}"
                self.logger.error(error_msg)
                await save_error_snapshot(f"clear_chat_overlay_timeout_{self.req_id}")
                raise Exception(error_msg)

            await self._check_disconnect(check_client_disconnected, "清空聊天 - 遮罩层出现后")
            self.logger.info(f"[{self.req_id}] 点击\"继续\"按钮 (在对话框中): {CLEAR_CHAT_CONFIRM_BUTTON_SELECTOR}")
            await confirm_button_locator.click(timeout=CLICK_TIMEOUT_MS)

        await self._check_disconnect(check_client_disconnected, "清空聊天 - 点击\"继续\"后")

        # 等待对话框消失
        max_retries_disappear = 3
        for attempt_disappear in range(max_retries_disappear):
            try:
                self.logger.info(f"[{self.req_id}] 等待清空聊天确认按钮/对话框消失 (尝试 {attempt_disappear + 1}/{max_retries_disappear})...")
                await expect_async(confirm_button_locator).to_be_hidden(timeout=CLEAR_CHAT_VERIFY_TIMEOUT_MS)
                await expect_async(overlay_locator).to_be_hidden(timeout=1000)
                self.logger.info(f"[{self.req_id}] ✅ 清空聊天确认对话框已成功消失。")
                break
            except TimeoutError:
                self.logger.warning(f"[{self.req_id}] ⚠️ 等待清空聊天确认对话框消失超时 (尝试 {attempt_disappear + 1}/{max_retries_disappear})。")
                if attempt_disappear < max_retries_disappear - 1:
                    await asyncio.sleep(1.0)
                    await self._check_disconnect(check_client_disconnected, f"清空聊天 - 重试消失检查 {attempt_disappear + 1} 前")
                    continue
                else:
                    error_msg = f"达到最大重试次数。清空聊天确认对话框未消失。请求 ID: {self.req_id}"
                    self.logger.error(error_msg)
                    await save_error_snapshot(f"clear_chat_dialog_disappear_timeout_{self.req_id}")
                    raise Exception(error_msg)
            except ClientDisconnectedError:
                self.logger.info(f"[{self.req_id}] 客户端在等待清空确认对话框消失时断开连接。")
                raise
            except Exception as other_err:
                self.logger.warning(f"[{self.req_id}] 等待清空确认对话框消失时发生其他错误: {other_err}")
                if attempt_disappear < max_retries_disappear - 1:
                    await asyncio.sleep(1.0)
                    continue
                else:
                    raise

            await self._check_disconnect(check_client_disconnected, f"清空聊天 - 消失检查尝试 {attempt_disappear + 1} 后")

    async def _verify_chat_cleared(self, check_client_disconnected: Callable):
        """验证聊天已清空"""
        last_response_container = self.page.locator(RESPONSE_CONTAINER_SELECTOR).last
        await asyncio.sleep(0.5)
        await self._check_disconnect(check_client_disconnected, "After Clear Post-Delay")
        try:
            await expect_async(last_response_container).to_be_hidden(timeout=CLEAR_CHAT_VERIFY_TIMEOUT_MS - 500)
            self.logger.info(f"[{self.req_id}] ✅ 聊天已成功清空 (验证通过 - 最后响应容器隐藏)。")
        except Exception as verify_err:
            self.logger.warning(f"[{self.req_id}] ⚠️ 警告: 清空聊天验证失败 (最后响应容器未隐藏): {verify_err}")
    
    async def submit_prompt(self, prompt: str,image_list: List, check_client_disconnected: Callable):
        """提交提示到页面。"""
        self.logger.info(f"[{self.req_id}] 填充并提交提示 ({len(prompt)} chars)...")
        prompt_textarea_locator = self.page.locator(PROMPT_TEXTAREA_SELECTOR)
        autosize_wrapper_locator = self.page.locator('ms-prompt-input-wrapper ms-autosize-textarea')
        submit_button_locator = self.page.locator(SUBMIT_BUTTON_SELECTOR)

        try:
            await expect_async(prompt_textarea_locator).to_be_visible(timeout=5000)
            await self._check_disconnect(check_client_disconnected, "After Input Visible")

            # 使用 JavaScript 填充文本
            await prompt_textarea_locator.evaluate(
                '''
                (element, text) => {
                    element.value = text;
                    element.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
                    element.dispatchEvent(new Event('change', { bubbles: true, cancelable: true }));
                }
                ''',
                prompt
            )
            await autosize_wrapper_locator.evaluate('(element, text) => { element.setAttribute("data-value", text); }', prompt)
            await self._check_disconnect(check_client_disconnected, "After Input Fill")

            # 上传
            if len(image_list) > 0:
                await self._upload_files_with_diagnostics(image_list, check_client_disconnected)

            # 等待发送按钮启用
            wait_timeout_ms_submit_enabled = 100000
            try:
                await self._check_disconnect(check_client_disconnected, "填充提示后等待发送按钮启用 - 前置检查")
                await expect_async(submit_button_locator).to_be_enabled(timeout=wait_timeout_ms_submit_enabled)
                self.logger.info(f"[{self.req_id}] ✅ 发送按钮已启用。")
            except Exception as e_pw_enabled:
                self.logger.error(f"[{self.req_id}] ❌ 等待发送按钮启用超时或错误: {e_pw_enabled}")
                await save_error_snapshot(f"submit_button_enable_timeout_{self.req_id}")
                raise

            await self._check_disconnect(check_client_disconnected, "After Submit Button Enabled")
            await asyncio.sleep(0.3)

            # 尝试使用快捷键提交
            submitted_successfully = await self._try_shortcut_submit(prompt_textarea_locator, check_client_disconnected)

            # 如果快捷键失败，使用按钮点击
            if not submitted_successfully:
                self.logger.info(f"[{self.req_id}] 快捷键提交失败，尝试点击提交按钮...")
                try:
                    await submit_button_locator.click(timeout=5000)
                    self.logger.info(f"[{self.req_id}] ✅ 提交按钮点击完成。")
                except Exception as click_err:
                    self.logger.error(f"[{self.req_id}] ❌ 提交按钮点击失败: {click_err}")
                    await save_error_snapshot(f"submit_button_click_fail_{self.req_id}")
                    raise

            await self._check_disconnect(check_client_disconnected, "After Submit")

        except Exception as e_input_submit:
            self.logger.error(f"[{self.req_id}] 输入和提交过程中发生错误: {e_input_submit}")
            if not isinstance(e_input_submit, ClientDisconnectedError):
                await save_error_snapshot(f"input_submit_error_{self.req_id}")
            raise

    async def _try_shortcut_submit(self, prompt_textarea_locator, check_client_disconnected: Callable) -> bool:
        """尝试使用快捷键提交"""
        import os
        try:
            # 检测操作系统
            host_os_from_launcher = os.environ.get('HOST_OS_FOR_SHORTCUT')
            is_mac_determined = False

            if host_os_from_launcher == "Darwin":
                is_mac_determined = True
            elif host_os_from_launcher in ["Windows", "Linux"]:
                is_mac_determined = False
            else:
                # 使用浏览器检测
                try:
                    user_agent_data_platform = await self.page.evaluate("() => navigator.userAgentData?.platform || ''")
                except Exception:
                    user_agent_string = await self.page.evaluate("() => navigator.userAgent || ''")
                    user_agent_string_lower = user_agent_string.lower()
                    if "macintosh" in user_agent_string_lower or "mac os x" in user_agent_string_lower:
                        user_agent_data_platform = "macOS"
                    else:
                        user_agent_data_platform = "Other"

                is_mac_determined = "mac" in user_agent_data_platform.lower()

            shortcut_modifier = "Meta" if is_mac_determined else "Control"
            shortcut_key = "Enter"

            self.logger.info(f"[{self.req_id}] 使用快捷键: {shortcut_modifier}+{shortcut_key}")

            await prompt_textarea_locator.focus(timeout=5000)
            await self._check_disconnect(check_client_disconnected, "After Input Focus")
            await asyncio.sleep(0.1)

            # 记录提交前的输入框内容，用于验证
            original_content = ""
            try:
                original_content = await prompt_textarea_locator.input_value(timeout=2000) or ""
            except Exception:
                # 如果无法获取原始内容，仍然尝试提交
                pass

            try:
                await self.page.keyboard.press(f'{shortcut_modifier}+{shortcut_key}')
            except Exception:
                # 尝试分步按键
                await self.page.keyboard.down(shortcut_modifier)
                await asyncio.sleep(0.05)
                await self.page.keyboard.press(shortcut_key)
                await asyncio.sleep(0.05)
                await self.page.keyboard.up(shortcut_modifier)

            await self._check_disconnect(check_client_disconnected, "After Shortcut Press")

            # 等待更长时间让提交完成
            await asyncio.sleep(2.0)

            # 多种方式验证提交是否成功
            submission_success = False

            try:
                # 方法1: 检查原始输入框是否清空
                current_content = await prompt_textarea_locator.input_value(timeout=2000) or ""
                if original_content and not current_content.strip():
                    self.logger.info(f"[{self.req_id}] 验证方法1: 输入框已清空，快捷键提交成功")
                    submission_success = True

                # 方法2: 检查提交按钮状态
                if not submission_success:
                    submit_button_locator = self.page.locator(SUBMIT_BUTTON_SELECTOR)
                    try:
                        is_disabled = await submit_button_locator.is_disabled(timeout=2000)
                        if is_disabled:
                            self.logger.info(f"[{self.req_id}] 验证方法2: 提交按钮已禁用，快捷键提交成功")
                            submission_success = True
                    except Exception:
                        pass

                # 方法3: 检查是否有响应容器出现
                if not submission_success:
                    try:
                        response_container = self.page.locator(RESPONSE_CONTAINER_SELECTOR)
                        container_count = await response_container.count()
                        if container_count > 0:
                            # 检查最后一个容器是否是新的
                            last_container = response_container.last
                            if await last_container.is_visible(timeout=1000):
                                self.logger.info(f"[{self.req_id}] 验证方法3: 检测到响应容器，快捷键提交成功")
                                submission_success = True
                    except Exception:
                        pass

            except Exception as verify_err:
                self.logger.warning(f"[{self.req_id}] 快捷键提交验证过程出错: {verify_err}")
                # 出错时假定提交成功，让后续流程继续
                submission_success = True

            if submission_success:
                self.logger.info(f"[{self.req_id}] ✅ 快捷键提交成功")
                return True
            else:
                self.logger.warning(f"[{self.req_id}] ⚠️ 快捷键提交验证失败")
                return False

        except Exception as shortcut_err:
            self.logger.warning(f"[{self.req_id}] 快捷键提交失败: {shortcut_err}")
            return False

    async def get_response(self, check_client_disconnected: Callable) -> str:
        """获取响应内容。"""
        self.logger.info(f"[{self.req_id}] 等待并获取响应...")

        try:
            # 等待响应容器出现
            response_container_locator = self.page.locator(RESPONSE_CONTAINER_SELECTOR).last
            response_element_locator = response_container_locator.locator(RESPONSE_TEXT_SELECTOR)

            self.logger.info(f"[{self.req_id}] 等待响应元素附加到DOM...")
            await expect_async(response_element_locator).to_be_attached(timeout=90000)
            await self._check_disconnect(check_client_disconnected, "获取响应 - 响应元素已附加")

            # 等待响应完成
            submit_button_locator = self.page.locator(SUBMIT_BUTTON_SELECTOR)
            edit_button_locator = self.page.locator(EDIT_MESSAGE_BUTTON_SELECTOR)
            input_field_locator = self.page.locator(PROMPT_TEXTAREA_SELECTOR)

            self.logger.info(f"[{self.req_id}] 等待响应完成...")
            completion_detected = await _wait_for_response_completion(
                self.page, input_field_locator, submit_button_locator, edit_button_locator, self.req_id, check_client_disconnected, None
            )

            if not completion_detected:
                self.logger.warning(f"[{self.req_id}] 响应完成检测失败，尝试获取当前内容")
            else:
                self.logger.info(f"[{self.req_id}] ✅ 响应完成检测成功")

            # 获取最终响应内容
            final_content = await _get_final_response_content(self.page, self.req_id, check_client_disconnected)

            if not final_content or not final_content.strip():
                self.logger.warning(f"[{self.req_id}] ⚠️ 获取到的响应内容为空")
                await save_error_snapshot(f"empty_response_{self.req_id}")
                # 不抛出异常，返回空内容让上层处理
                return ""

            self.logger.info(f"[{self.req_id}] ✅ 成功获取响应内容 ({len(final_content)} chars)")
            return final_content

        except Exception as e:
            self.logger.error(f"[{self.req_id}] ❌ 获取响应时出错: {e}")
            if not isinstance(e, ClientDisconnectedError):
                await save_error_snapshot(f"get_response_error_{self.req_id}")
            raise

    async def _upload_files_with_diagnostics(self, image_list, check_client_disconnected):
        """带诊断信息的文件上传函数"""
        self.logger.info(f"[{self.req_id}] 开始文件上传诊断，文件数量: {len(image_list)}")

        try:
            # 1. 诊断页面状态
            await self._diagnose_upload_capability()

            # 2. 查找并点击功能按钮
            function_btn_found = await self._find_and_click_function_button()
            if not function_btn_found:
                self.logger.error(f"[{self.req_id}] ❌ 无法找到功能按钮，可能此账号不支持文件上传")
                return

            # 3. 尝试多种上传方式
            upload_success = await self._try_multiple_upload_methods(image_list)

            if upload_success:
                self.logger.info(f"[{self.req_id}] ✅ 文件上传成功")
                # 4. 处理版权确认
                await self._handle_copyright_acknowledgement()
            else:
                self.logger.error(f"[{self.req_id}] ❌ 所有上传方法都失败")

        except Exception as e:
            self.logger.error(f"[{self.req_id}] ❌ 文件上传过程中发生错误: {e}")
            # 不抛出异常，让流程继续，只是没有文件上传

    async def _diagnose_upload_capability(self):
        """诊断当前页面的上传能力"""
        self.logger.info(f"[{self.req_id}] 开始诊断页面上传能力...")

        # 检查AI Studio的上传相关元素
        selectors_to_check = [
            # AI Studio主要的附件按钮（⊕ 按钮）
            ('AI Studio插入按钮', 'button[aria-label="Insert assets such as images, videos, files, or audio"]'),

            # 文件输入元素（最重要的）
            ('文件输入', 'input[type="file"]'),
            ('图片输入', 'input[accept*="image"]'),

            # AI Studio菜单中的选项（点击附件按钮后出现）
            ('Upload File菜单项', 'div[role="menuitem"]:has-text("Upload File")'),
            ('My Drive菜单项', 'div[role="menuitem"]:has-text("My Drive")'),
            ('Camera菜单项', 'div[role="menuitem"]:has-text("Camera")'),
            ('Sample Media菜单项', 'div[role="menuitem"]:has-text("Sample Media")'),

            # 备用选择器
            ('上传按钮', 'button[aria-label="Upload File"]'),
            ('附件按钮', 'button[aria-label="Attach files"]')
        ]

        found_elements = []
        detailed_info = []

        for name, selector in selectors_to_check:
            try:
                locator = self.page.locator(selector)
                count = await locator.count()
                if count > 0:
                    found_elements.append(f"{name}: {selector} (count: {count})")

                    # 获取第一个元素的详细信息
                    try:
                        first_element = locator.first
                        is_visible = await first_element.is_visible()
                        is_enabled = await first_element.is_enabled()

                        # 获取元素的文本内容
                        text_content = ""
                        try:
                            text_content = await first_element.text_content() or ""
                            text_content = text_content.strip()[:50]  # 限制长度
                        except Exception:
                            pass

                        detailed_info.append(f"  {name}: visible={is_visible}, enabled={is_enabled}, text='{text_content}'")
                    except Exception:
                        pass

            except Exception:
                pass

        if found_elements:
            self.logger.info(f"[{self.req_id}] 找到的上传相关元素:")
            for element in found_elements:
                self.logger.info(f"[{self.req_id}]   {element}")

            if detailed_info:
                self.logger.info(f"[{self.req_id}] 元素详细状态:")
                for info in detailed_info:
                    self.logger.info(f"[{self.req_id}] {info}")
        else:
            self.logger.warning(f"[{self.req_id}] ⚠️ 未找到任何上传相关元素，此账号可能不支持文件上传")

        # 检查页面URL和标题
        try:
            current_url = self.page.url
            page_title = await self.page.title()
            self.logger.info(f"[{self.req_id}] 页面信息 - URL: {current_url}")
            self.logger.info(f"[{self.req_id}] 页面信息 - Title: {page_title}")
        except Exception:
            pass

        # AI Studio对所有账号提供统一的功能，无需检查账号限制



    async def _find_and_click_function_button(self):
        """查找并点击功能按钮 - 针对AI Studio优化"""
        # 针对AI Studio的功能按钮选择器（基于实际界面）
        function_button_selectors = [
            # AI Studio主要的附件按钮（⊕ 按钮）
            'button[aria-label="Insert assets such as images, videos, files, or audio"]',
            'button[aria-label="Add content"]',
            'button[aria-label="Insert content"]',
            'button[title="Insert assets such as images, videos, files, or audio"]',

            # 可能的其他变体
            'button[aria-label="Attach files"]',
            'button[aria-label="Upload files"]',
            'button[aria-label="Add files"]',
            'button[title="Insert assets"]',
            'button[data-testid="attach-button"]',
            'button[data-testid="file-upload-button"]',

            # 通用选择器
            'button[aria-label="Add media"]',
            'button[title="Attach"]'
        ]

        for selector in function_button_selectors:
            try:
                locator = self.page.locator(selector)
                count = await locator.count()
                if count > 0:
                    # 检查按钮状态
                    first_btn = locator.first
                    is_visible = await first_btn.is_visible()
                    is_enabled = await first_btn.is_enabled()

                    self.logger.info(f"[{self.req_id}] 找到功能按钮: {selector} (visible: {is_visible}, enabled: {is_enabled})")

                    if is_enabled:
                        await first_btn.click()
                        self.logger.info(f"[{self.req_id}] ✅ 成功点击功能按钮")
                        await asyncio.sleep(0.5)  # 等待菜单展开，AI Studio通常响应较快
                        return True
                    else:
                        self.logger.warning(f"[{self.req_id}] 功能按钮不可用: {selector}")

            except Exception as e:
                self.logger.debug(f"[{self.req_id}] 尝试点击 {selector} 失败: {e}")

        self.logger.error(f"[{self.req_id}] ❌ 无法找到任何可用的功能按钮")
        return False

    async def _try_multiple_upload_methods(self, image_list):
        """尝试多种上传方法，按可靠性排序"""
        self.logger.info(f"[{self.req_id}] 尝试多种上传方法...")

        # 方法1: 直接查找input[type="file"] - 最可靠的方法
        self.logger.info(f"[{self.req_id}] === 尝试方法1: 直接文件输入 ===")
        if await self._try_direct_file_input(image_list):
            return True

        # 方法2: 标准的文件选择器方法
        self.logger.info(f"[{self.req_id}] === 尝试方法2: 标准文件选择器 ===")
        if await self._try_standard_file_chooser(image_list):
            return True

        # 方法3: 拖拽上传
        self.logger.info(f"[{self.req_id}] === 尝试方法3: 拖拽上传 ===")
        if await self._try_drag_drop_upload(image_list):
            return True

        self.logger.error(f"[{self.req_id}] ❌ 所有上传方法都失败了")
        return False

    async def _try_standard_file_chooser(self, image_list):
        """标准文件选择器方法"""
        self.logger.info(f"[{self.req_id}] 尝试标准文件选择器方法...")

        # 针对AI Studio的上传按钮选择器（基于实际界面）
        upload_button_selectors = [
            # AI Studio菜单中的"Upload File"选项
            'div[role="menuitem"]:has-text("Upload File")',
            'button[role="menuitem"]:has-text("Upload File")',
            'mat-menu-item:has-text("Upload File")',
            '[aria-label*="Upload File"]',

            # 原有的主要选择器
            UPLOAD_BUTTON_SELECTOR,  # 'button[aria-label="Upload File"]'

            # 其他可能的选择器
            'button[aria-label="Upload from computer"]',
            'button[aria-label="Upload files"]',
            'button[aria-label="Choose files"]',
            'button[aria-label="Browse files"]',

            # 通用菜单项选择器
            'div[role="menuitem"]:has-text("Upload")',
            'button[role="menuitem"]:has-text("Upload")',
            'mat-menu-item:has-text("Upload")',

            # 通用按钮选择器
            'button[aria-label="Upload"]',
            'button[title="Upload File"]',
            'button:has-text("Upload File")',
            'button:has-text("Upload")',
            '[data-testid="upload-button"]',
            'button:has-text("Choose file")',
            'button:has-text("Browse")',
            'button:has-text("Select files")'
        ]

        for selector in upload_button_selectors:
            try:
                upload_btn_locator = self.page.locator(selector)
                count = await upload_btn_locator.count()
                if count == 0:
                    self.logger.debug(f"[{self.req_id}] 选择器 {selector} 未找到元素")
                    continue

                self.logger.info(f"[{self.req_id}] 找到上传按钮: {selector} (数量: {count})")

                # 检查按钮是否可见和可点击
                first_btn = upload_btn_locator.first
                is_visible = await first_btn.is_visible()
                is_enabled = await first_btn.is_enabled()

                self.logger.info(f"[{self.req_id}] 按钮状态 - visible: {is_visible}, enabled: {is_enabled}")

                if not is_enabled:
                    self.logger.warning(f"[{self.req_id}] 按钮不可用，跳过: {selector}")
                    continue

                # 尝试两种方法：先监听再点击 vs 先点击再监听
                success = await self._try_file_chooser_method1(first_btn, image_list, selector)
                if success:
                    return True

                success = await self._try_file_chooser_method2(first_btn, image_list, selector)
                if success:
                    return True

            except Exception as e:
                self.logger.warning(f"[{self.req_id}] 使用选择器 {selector} 失败: {e}")

        return False

    async def _try_file_chooser_method1(self, button_locator, image_list, selector):
        """方法1: 先监听文件选择器，再点击按钮"""
        try:
            self.logger.info(f"[{self.req_id}] 方法1 - 先监听再点击: {selector}")

            async with self.page.expect_file_chooser(timeout=10000) as fc_info:
                await button_locator.click()
                self.logger.info(f"[{self.req_id}] 已点击按钮，等待文件选择器...")

            file_chooser = await fc_info.value
            self.logger.info(f"[{self.req_id}] ✅ 文件选择器已出现 (方法1)")

            await file_chooser.set_files(image_list)
            self.logger.info(f"[{self.req_id}] ✅ 已设置文件: {image_list}")

            await asyncio.sleep(1.0)
            return True

        except Exception as e:
            self.logger.debug(f"[{self.req_id}] 方法1失败: {e}")
            return False

    async def _try_file_chooser_method2(self, button_locator, image_list, selector):
        """方法2: 先点击按钮，再尝试监听文件选择器"""
        try:
            self.logger.info(f"[{self.req_id}] 方法2 - 先点击再监听: {selector}")

            # 先点击按钮
            await button_locator.click()
            self.logger.info(f"[{self.req_id}] 已点击按钮")

            # 短暂等待
            await asyncio.sleep(0.2)

            # 再尝试监听文件选择器
            async with self.page.expect_file_chooser(timeout=5000) as fc_info:
                # 可能需要再次点击或者等待
                pass

            file_chooser = await fc_info.value
            self.logger.info(f"[{self.req_id}] ✅ 文件选择器已出现 (方法2)")

            await file_chooser.set_files(image_list)
            self.logger.info(f"[{self.req_id}] ✅ 已设置文件: {image_list}")

            await asyncio.sleep(1.0)
            return True

        except Exception as e:
            self.logger.debug(f"[{self.req_id}] 方法2失败: {e}")
            return False

    async def _try_direct_file_input(self, image_list):
        """直接查找文件输入元素 - 这是最可靠的方法"""
        self.logger.info(f"[{self.req_id}] 尝试直接文件输入方法...")

        try:
            # 查找所有可能的文件输入元素
            file_input_selectors = [
                'input[type="file"]',
                'input[accept*="image"]',
                'input[accept*="file"]'
            ]

            all_file_inputs = []
            for selector in file_input_selectors:
                try:
                    inputs = await self.page.locator(selector).all()
                    all_file_inputs.extend(inputs)
                except Exception:
                    pass

            if not all_file_inputs:
                self.logger.info(f"[{self.req_id}] 未找到任何文件输入元素")
                return False

            self.logger.info(f"[{self.req_id}] 找到 {len(all_file_inputs)} 个文件输入元素")

            for i, file_input in enumerate(all_file_inputs):
                try:
                    # 获取元素的详细信息
                    is_visible = await file_input.is_visible()
                    is_enabled = await file_input.is_enabled()

                    # 获取accept属性
                    accept_attr = ""
                    try:
                        accept_attr = await file_input.get_attribute('accept') or ""
                    except Exception:
                        pass

                    # 获取元素的位置信息
                    bounding_box = None
                    try:
                        bounding_box = await file_input.bounding_box()
                    except Exception:
                        pass

                    self.logger.info(f"[{self.req_id}] 文件输入元素 {i}: visible={is_visible}, enabled={is_enabled}, accept='{accept_attr}', bbox={bounding_box}")

                    # 尝试上传文件，即使元素不可见（很多文件输入都是隐藏的）
                    if is_enabled:
                        try:
                            await file_input.set_input_files(image_list)
                            self.logger.info(f"[{self.req_id}] ✅ 通过文件输入元素 {i} 上传成功")

                            # 等待文件处理
                            await asyncio.sleep(2.0)

                            # 验证上传是否成功（检查是否有文件名显示等）
                            if await self._verify_file_upload_success():
                                return True
                            else:
                                self.logger.info(f"[{self.req_id}] 文件输入元素 {i} 上传验证失败，继续尝试其他元素")

                        except Exception as upload_err:
                            self.logger.debug(f"[{self.req_id}] 文件输入元素 {i} 上传失败: {upload_err}")

                except Exception as e:
                    self.logger.debug(f"[{self.req_id}] 检查文件输入元素 {i} 时出错: {e}")

        except Exception as e:
            self.logger.warning(f"[{self.req_id}] 直接文件输入方法失败: {e}")

        return False

    async def _verify_file_upload_success(self):
        """验证文件上传是否成功 - 针对AI Studio优化"""
        try:
            # 检查AI Studio特定的上传成功指示器
            ai_studio_success_indicators = [
                # AI Studio可能显示上传的图片
                'img[src*="blob:"]',  # 上传的图片通常会显示为blob URL
                'img[src*="data:"]',  # 或者data URL

                # AI Studio可能的文件显示元素
                '.uploaded-image',
                '.attachment-preview',
                '[data-testid="uploaded-image"]',
                '[data-testid="attachment"]',

                # 通用的上传成功指示器
                '.file-name',
                '.uploaded-file',
                '[data-testid="uploaded-file"]',
                '.file-preview',
                '.attachment',

                # AI Studio输入区域可能的变化
                'ms-prompt-input-wrapper img',  # 输入区域内的图片
                '.prompt-input img',

                # 可能的文件标签或芯片
                'mat-chip',
                '.file-chip',
                '.attachment-chip'
            ]

            for indicator in ai_studio_success_indicators:
                try:
                    count = await self.page.locator(indicator).count()
                    if count > 0:
                        self.logger.info(f"[{self.req_id}] 发现上传成功指示器: {indicator} (count: {count})")

                        # 对于图片，尝试获取更多信息
                        if 'img' in indicator:
                            try:
                                img_src = await self.page.locator(indicator).first.get_attribute('src')
                                if img_src:
                                    self.logger.info(f"[{self.req_id}] 图片源: {img_src[:100]}...")
                            except Exception:
                                pass

                        return True
                except Exception:
                    pass

            # 检查输入区域是否有变化
            try:
                # 检查输入区域的子元素数量是否增加
                input_wrapper = self.page.locator('ms-prompt-input-wrapper')
                if await input_wrapper.count() > 0:
                    child_count = await input_wrapper.locator('*').count()
                    self.logger.debug(f"[{self.req_id}] 输入区域子元素数量: {child_count}")
            except Exception:
                pass

            # 等待一下再检查
            await asyncio.sleep(1.0)
            return False

        except Exception as e:
            self.logger.debug(f"[{self.req_id}] 验证上传成功时出错: {e}")
            return False

    async def _try_drag_drop_upload(self, image_list):
        """尝试拖拽上传"""
        self.logger.info(f"[{self.req_id}] 尝试拖拽上传方法...")

        try:
            # 查找可能的拖拽区域
            drop_zones = [
                '[data-testid="drop-zone"]',
                '.drop-zone',
                '.upload-area',
                'textarea',  # 有些网站支持直接拖拽到文本区域
            ]

            for zone_selector in drop_zones:
                try:
                    zone_locator = self.page.locator(zone_selector)
                    count = await zone_locator.count()
                    if count > 0:
                        self.logger.info(f"[{self.req_id}] 尝试拖拽到: {zone_selector}")

                        # 模拟文件拖拽
                        await zone_locator.first.dispatch_event('dragover')
                        await zone_locator.first.dispatch_event('drop', {
                            'dataTransfer': {
                                'files': image_list
                            }
                        })

                        await asyncio.sleep(2.0)
                        self.logger.info(f"[{self.req_id}] ✅ 拖拽上传尝试完成")
                        return True

                except Exception as e:
                    self.logger.debug(f"[{self.req_id}] 拖拽到 {zone_selector} 失败: {e}")

        except Exception as e:
            self.logger.warning(f"[{self.req_id}] 拖拽上传方法失败: {e}")

        return False

    async def _handle_copyright_acknowledgement(self):
        """处理版权确认"""
        try:
            acknow_selectors = [
                'button[aria-label="Agree to the copyright acknowledgement"]',
                'button:has-text("Agree")',
                'button:has-text("I agree")',
                'button:has-text("Accept")',
                '[data-testid="copyright-agree"]'
            ]

            for selector in acknow_selectors:
                try:
                    acknow_btn_locator = self.page.locator(selector)
                    count = await acknow_btn_locator.count()
                    if count > 0:
                        self.logger.info(f"[{self.req_id}] 找到版权确认按钮: {selector}")
                        await acknow_btn_locator.first.click()
                        self.logger.info(f"[{self.req_id}] ✅ 版权确认完成")
                        return
                except Exception as e:
                    self.logger.debug(f"[{self.req_id}] 版权确认按钮 {selector} 点击失败: {e}")

            self.logger.info(f"[{self.req_id}] 未找到版权确认按钮，可能不需要确认")

        except Exception as e:
            self.logger.warning(f"[{self.req_id}] 处理版权确认时出错: {e}")