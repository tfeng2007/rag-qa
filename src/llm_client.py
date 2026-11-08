from dotenv import load_dotenv
load_dotenv()                    # 读取 .env，把里面的键值塞进环境变量

import os
import anthropic

client = anthropic.Anthropic()   # 自动读 ANTHROPIC_API_KEY，不用手动传
MODEL = os.environ.get("MODEL", "claude-sonnet-4-6")   # 读 MODEL，没有就用默认