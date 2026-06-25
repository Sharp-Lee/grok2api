import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


def _load_script_module():
    file_path = Path(__file__).resolve().parents[1] / "scripts/generate_image.py"
    spec = importlib.util.spec_from_file_location("generate_image_script", file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


script = _load_script_module()


class GenerateImageScriptTests(unittest.TestCase):
    def test_chat_payload_uses_message_history(self):
        messages = [
            {"role": "user", "content": "画一只猫"},
            {"role": "assistant", "content": "![image](/v1/files/image?id=abc)"},
            {"role": "user", "content": "保持风格，改成狗"},
        ]

        endpoint, payload = script.build_chat_payload(
            model="grok-4.20-fast",
            messages=messages,
            n=1,
            size="1024x1024",
            response_format="url",
        )

        self.assertEqual(endpoint, "chat")
        self.assertEqual(payload["messages"], messages)

    def test_chat_payload_can_include_image_tool(self):
        tools = [script.build_image_tool_schema()]

        _, payload = script.build_chat_payload(
            model="grok-4.3-console",
            messages=[{"role": "user", "content": "生成一张照片"}],
            n=1,
            size="1024x1024",
            response_format="url",
            tools=tools,
            tool_choice="auto",
        )

        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"], "auto")

    def test_extract_tool_calls_from_chat_response(self):
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "generate_photo",
                                    "arguments": '{"prompt":"雨夜东京街头","n":1}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        calls = script.extract_tool_calls(response)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "generate_photo")

    def test_parse_image_tool_arguments(self):
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "generate_photo",
                "arguments": '{"prompt":"雨夜东京街头","n":2,"size":"1280x720"}',
            },
        }

        args = script.parse_tool_call_arguments(tool_call)

        self.assertEqual(args["prompt"], "雨夜东京街头")
        self.assertEqual(args["n"], 2)
        self.assertEqual(args["size"], "1280x720")

    def test_interactive_enables_image_tool_by_default(self):
        args = SimpleNamespace(
            agent=False,
            interactive=True,
            enable_image_tool=False,
            disable_image_tool=False,
            disable_auto_image_fallback=False,
        )

        script.configure_interactive_agent_flags(args)

        self.assertTrue(args.interactive)
        self.assertTrue(args.enable_image_tool)
        self.assertTrue(args.auto_image_fallback)

    def test_interactive_can_disable_image_tool(self):
        args = SimpleNamespace(
            agent=False,
            interactive=True,
            enable_image_tool=False,
            disable_image_tool=True,
            disable_auto_image_fallback=False,
        )

        script.configure_interactive_agent_flags(args)

        self.assertTrue(args.interactive)
        self.assertFalse(args.enable_image_tool)
        self.assertTrue(args.auto_image_fallback)

    def test_interactive_can_disable_auto_image_fallback(self):
        args = SimpleNamespace(
            agent=False,
            interactive=True,
            enable_image_tool=False,
            disable_image_tool=False,
            disable_auto_image_fallback=True,
        )

        script.configure_interactive_agent_flags(args)

        self.assertFalse(args.auto_image_fallback)

    def test_direct_image_command_extracts_prompt(self):
        self.assertEqual(
            script.direct_image_prompt_from_command("/image 沙滩边的美女照片"),
            "沙滩边的美女照片",
        )
        self.assertEqual(
            script.direct_image_prompt_from_command("/photo cinematic beach portrait"),
            "cinematic beach portrait",
        )

    def test_image_intent_detects_photo_prompt(self):
        self.assertTrue(script.looks_like_image_request("沙滩边的美女照片"))
        self.assertTrue(script.looks_like_image_request("draw a cinematic beach portrait"))
        self.assertFalse(script.looks_like_image_request("你觉得这个想法怎么样"))

    def test_append_assistant_response_updates_history(self):
        history = [{"role": "user", "content": "画一张绿色勾选图标"}]
        response = {
            "choices": [
                {"message": {"content": "![image](/v1/files/image?id=abc)"}}
            ]
        }

        content = script.append_assistant_response(history, response)

        self.assertEqual(content, "![image](/v1/files/image?id=abc)")
        self.assertEqual(
            history,
            [
                {"role": "user", "content": "画一张绿色勾选图标"},
                {"role": "assistant", "content": "![image](/v1/files/image?id=abc)"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
