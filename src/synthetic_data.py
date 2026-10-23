"""生成合成商品与交易数据（模拟"从外部采集 + 下单"得到的数据）。

设计原则
--------
1. **绝不修改 ``data/raw/`` 下的原始数据**。生成结果写到 ``data/generated/``，
   两份数据各自独立存在，互不影响。
2. **清洗规则完全复用**。生成数据与原始数据是一模一样的表结构
   （transactions 8 列 / products 4 列），因此阶段一那套三级清洗漏斗
   可以原样套用，不需要为它改任何规则。
3. **生成的是"像真的"而不是"随机的"**：购物篮按主题聚集（同类目 / 相关类目成组出现），
   商品热度服从长尾分布，并**按真实比例注入脏数据**（缺用户 ID、退货、异常价格），
   这样清洗漏斗不是空跑。
4. 全流程用固定随机种子，结果可复现。

v2 新增（为支撑"推荐首页 · 季节档期"）
--------------------------------------
5. **真实季节性**：每个类目带一组"月度权重"（``months``），生成日期时按权重抽，
   于是"9 月开学季数码/文具卖得好""11 月双 11""12 月圣诞"在数据里**真的成立**，
   而不是只写在档期日历里。见 :data:`MONTH_PRESETS`。
6. **品牌化 3C 商品**：手机、笔记本、平板、耳机、手表、相机、游戏机这些类目用
   **真实品牌 + 型号 + 规格**命名（`苹果 iPhone 15 Pro 256GB`、
   `华为 Mate 60 Pro 12GB+512GB`），不再只有"磁吸充电器 Pro版"这类通用名。
7. **补充原本很薄的类目**：`文具用品`（原始只有 4 个商品）、`办公用品`（3 个）
   在合成数据里各补 200 个，另新增 `宿舍用品`，让"开学季"档期在商品库里真的有货。
8. **人物画像**：内置 :data:`PERSONAS`，按"剧本"生成整年连贯的购买记录
   （例如大学生 QZJ 的入学采购、618 换手机、开学季买平板、双 11 囤耳机），
   供推荐首页做个性化演示。
9. **客户品类偏好**：真实用户有稳定的品类偏好，不是每次随机逛。生成时给每位客户
   分配 1~3 个偏好类目（原始数据的真实客户直接沿用**他在原始数据里的真实偏好**），
   72% 的订单落在偏好类目内。没有这一层，用户-商品矩阵就是噪声，
   协同过滤会输给热门基线 —— 这是实测踩过的坑。

为什么这么设计
--------------
原始数据只有 4070 个商品、22 个类目，且**均价中位数只有 1.95 元**，
导致"100 元以内"这类价格条件几乎不过滤。补充数据刻意覆盖更高的价格带
（几十到上万元）与全新的类目，让价格筛选、类目筛选与季节档期真正起作用。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 月度权重预设
# 每个预设是 {月份: 相对权重}，没写到的月份按 1.0 处理。
# 这套权重是**业务假设**（开学季、618、双 11、圣诞、换季），
# 生成出来的数据因此带有真实季节性 —— 推荐首页的季节指数就是从这份数据算出来的。
MONTH_PRESETS: dict[str, dict[int, float]] = {
    # 3C 数码：8~9 月开学季 + 6 月 618 + 11 月双 11
    "back_to_school": {
        1: 0.9, 2: 0.8, 3: 1.0, 4: 0.9, 5: 1.0, 6: 1.5,
        7: 1.2, 8: 1.9, 9: 2.4, 10: 1.1, 11: 2.0, 12: 1.2,
    },
    # 学生文具 / 教辅 / 宿舍：开学季更极端
    "campus": {
        1: 1.0, 2: 0.9, 3: 1.3, 4: 1.0, 5: 1.0, 6: 1.2,
        7: 1.0, 8: 2.2, 9: 2.6, 10: 1.0, 11: 1.3, 12: 0.9,
    },
    # 家电：618 与双 11 两个大促 + 年末
    "appliance": {
        1: 1.1, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.1, 6: 1.6,
        7: 1.2, 8: 1.0, 9: 0.9, 10: 1.1, 11: 1.8, 12: 1.3,
    },
    # 户外运动：春秋两季
    "outdoor": {
        1: 0.8, 2: 0.9, 3: 1.4, 4: 1.5, 5: 1.3, 6: 1.1,
        7: 1.4, 8: 1.2, 9: 1.3, 10: 1.0, 11: 0.9, 12: 0.8,
    },
    # 美妆 / 珠宝等礼赠品：情人节、母亲节、双 11、圣诞
    "gift_beauty": {
        1: 1.0, 2: 1.7, 3: 1.2, 4: 1.0, 5: 1.4, 6: 1.2,
        7: 1.0, 8: 0.9, 9: 1.1, 10: 1.1, 11: 1.6, 12: 1.4,
    },
    # 母婴：大促 + 年末
    "family": {
        1: 1.0, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.2, 6: 1.2,
        7: 1.0, 8: 1.0, 9: 1.1, 10: 1.0, 11: 1.4, 12: 1.1,
    },
    # 出行 / 汽车：春游与暑期
    "travel": {
        1: 0.9, 2: 0.9, 3: 1.2, 4: 1.2, 5: 1.1, 6: 1.0,
        7: 1.4, 8: 1.4, 9: 1.0, 10: 1.0, 11: 1.2, 12: 1.0,
    },
    # 食品：春节、中秋、圣诞
    "food": {
        1: 1.6, 2: 1.2, 3: 0.9, 4: 0.9, 5: 1.0, 6: 1.2,
        7: 1.0, 8: 1.0, 9: 1.3, 10: 1.0, 11: 1.1, 12: 1.5,
    },
    # 服饰鞋包：换季与开学季
    "fashion": {
        1: 0.9, 2: 0.9, 3: 1.2, 4: 1.1, 5: 1.0, 6: 1.0,
        7: 1.1, 8: 1.5, 9: 1.6, 10: 1.1, 11: 1.4, 12: 1.0,
    },
    # ---- 原始类目的"高端补充商品"用这几套权重 ----
    # 家居焕新：新年、春季、双 11~圣诞
    "home_decor": {
        1: 1.3, 2: 0.9, 3: 1.3, 4: 1.0, 5: 1.0, 6: 0.9,
        7: 0.9, 8: 1.0, 9: 1.2, 10: 1.3, 11: 1.4, 12: 1.3,
    },
    # 节庆装饰 / 派对 / 玩具 / 手工：Q4 爆发
    "festive": {
        1: 1.0, 2: 0.9, 3: 0.9, 4: 0.9, 5: 1.0, 6: 0.9,
        7: 0.9, 8: 1.0, 9: 1.1, 10: 1.3, 11: 1.8, 12: 2.2,
    },
    # 洗浴 / 浴室：夏季
    "summer_bath": {
        1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.2, 6: 1.5,
        7: 1.6, 8: 1.4, 9: 1.0, 10: 0.9, 11: 0.9, 12: 0.8,
    },
    # 园艺 / 花园：春季
    "gardening": {
        1: 0.8, 2: 0.9, 3: 1.5, 4: 1.7, 5: 1.4, 6: 1.1,
        7: 1.0, 8: 0.9, 9: 0.9, 10: 0.9, 11: 0.9, 12: 0.8,
    },
    # 工具：装修旺季（春、秋）
    "tools": {
        1: 1.0, 2: 0.9, 3: 1.2, 4: 1.2, 5: 1.1, 6: 1.0,
        7: 1.0, 8: 1.0, 9: 1.2, 10: 1.1, 11: 1.0, 12: 1.0,
    },
}


def _months(preset: str) -> dict[int, float]:
    """取一份月度权重预设（返回副本，避免调用方改到全局）。"""
    return dict(MONTH_PRESETS[preset])


# ---------------------------------------------------------------- 类目定义
# 两种命名方式：
#   A. brands       品牌 × 型号 × 规格   →  "苹果 iPhone 15 Pro 256GB"
#   B. nouns+themes 名词 × 系列词 × 规格 →  "磁吸充电器 Pro版"
# months 决定这个类目的月度权重（真实季节性的来源）。
CATEGORY_SPECS: dict[str, dict[str, Any]] = {
    # ==================== 3C / 数码（v2 重点扩充） ====================
    "手机": {
        "prefix": "PH",
        "price": (899, 14999),
        "months": _months("back_to_school"),
        "brands": [
            ("苹果", "Apple", ["iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone 15", "iPhone 14"]),
            ("华为", "HUAWEI", ["Mate 60 Pro+", "Mate 60 Pro", "P60 Art", "nova 12 Pro"]),
            ("小米", "Xiaomi", ["14 Ultra", "14 Pro", "Redmi K70 Pro", "Redmi Note 13"]),
            ("OPPO", "OPPO", ["Find X7 Ultra", "Reno11 Pro", "K11"]),
            ("vivo", "vivo", ["X100 Pro", "S18 Pro", "iQOO 12"]),
            ("荣耀", "Honor", ["Magic6 Pro", "100 Pro", "X50"]),
            ("一加", "OnePlus", ["12 Pro", "Ace 3"]),
            ("三星", "Samsung", ["Galaxy S24 Ultra", "Galaxy S23", "Galaxy A55"]),
            ("真我", "realme", ["GT5 Pro", "真我 12 Pro"]),
            ("魅族", "MEIZU", ["21 Pro", "20 Infinity"]),
        ],
        "variants": [
            ("8GB+256GB", "8GB+256GB"), ("12GB+256GB", "12GB+256GB"),
            ("12GB+512GB", "12GB+512GB"), ("16GB+512GB", "16GB+512GB"),
            ("16GB+1TB", "16GB+1TB"), ("24GB+1TB", "24GB+1TB"),
        ],
    },
    "笔记本电脑": {
        "prefix": "LP",
        "price": (2999, 24999),
        "months": _months("back_to_school"),
        "brands": [
            ("苹果", "Apple", ["MacBook Air M3 13英寸", "MacBook Air M3 15英寸", "MacBook Pro 14 M3"]),
            ("联想", "Lenovo", ["小新 Pro 16", "拯救者 Y9000P", "ThinkBook 14+", "YOGA Air 14"]),
            ("华为", "HUAWEI", ["MateBook X Pro", "MateBook 14s", "MateBook D 16"]),
            ("戴尔", "DELL", ["XPS 13", "灵越 14 Plus", "游匣 G15"]),
            ("华硕", "ASUS", ["天选 5", "无畏 Pro 15", "灵耀 14"]),
            ("惠普", "HP", ["暗影精灵 10", "星 Book Pro 14", "战 66"]),
            ("小米", "Xiaomi", ["Redmi Book Pro 16", "小米笔记本 Air 13"]),
            ("宏碁", "acer", ["非凡 Go 14", "掠夺者 擎 Neo"]),
            ("微星", "MSI", ["泰坦 18 HX", "绝影 14 Studio"]),
            ("机械革命", "MECHREVO", ["极光 Pro", "蛟龙 16K"]),
            ("神舟", "HASEE", ["战神 Z8", "优雅 X5"]),
        ],
        "variants": [
            ("16GB+512GB", "16GB+512GB"), ("16GB+1TB", "16GB+1TB"),
            ("32GB+1TB", "32GB+1TB"), ("32GB+2TB", "32GB+2TB"),
            ("i5 16GB+512GB", "i5 16GB+512GB"), ("i7 32GB+1TB", "i7 32GB+1TB"),
        ],
    },
    "平板电脑": {
        "prefix": "TB",
        "price": (999, 9999),
        "months": _months("back_to_school"),
        "brands": [
            ("苹果", "Apple", ["iPad Pro 11", "iPad Air 5", "iPad 10", "iPad mini 6"]),
            ("华为", "HUAWEI", ["MatePad Pro 13.2", "MatePad 11.5", "MatePad SE"]),
            ("小米", "Xiaomi", ["平板 6 Pro", "平板 6S Pro", "Redmi Pad Pro"]),
            ("荣耀", "Honor", ["平板 V8 Pro", "平板 9", "平板 X8 Pro"]),
            ("联想", "Lenovo", ["小新 Pad Pro 12.7", "拯救者 Y700", "小新 Pad 11"]),
            ("三星", "Samsung", ["Galaxy Tab S9", "Galaxy Tab A9+"]),
            ("OPPO", "OPPO", ["Pad 3 Pro", "Pad Air 2"]),
        ],
        "variants": [
            ("8GB+128GB", "8GB+128GB"), ("8GB+256GB", "8GB+256GB"),
            ("12GB+256GB", "12GB+256GB"), ("16GB+512GB", "16GB+512GB"),
            ("WiFi版", "WiFi"), ("插卡版", "Cellular"),
        ],
    },
    "智能穿戴": {
        "prefix": "WR",
        "price": (149, 5999),
        "months": _months("back_to_school"),
        "brands": [
            ("苹果", "Apple", ["Watch Series 9", "Watch SE", "Watch Ultra 2"]),
            ("华为", "HUAWEI", ["WATCH GT4", "WATCH 4 Pro", "手环 9"]),
            ("小米", "Xiaomi", ["手环 8 Pro", "手表 S3", "手表 2"]),
            ("荣耀", "Honor", ["手表 4", "手环 8", "手表 GS 3"]),
            ("佳明", "Garmin", ["Forerunner 165", "Fenix 7", "Venu 3"]),
            ("三星", "Samsung", ["Galaxy Watch6", "Galaxy Fit3"]),
            ("华米", "Amazfit", ["GTR 4", "T-Rex 3"]),
        ],
        "variants": [
            ("41mm", "41mm"), ("45mm", "45mm"), ("标准版", "Standard"),
            ("NFC版", "NFC"), ("运动版", "Sport"), ("真皮表带款", "Leather Strap"),
        ],
    },
    "耳机音响": {
        "prefix": "AU",
        "price": (99, 3999),
        "months": _months("back_to_school"),
        "brands": [
            ("苹果", "Apple", ["AirPods Pro 2", "AirPods 3", "AirPods Max"]),
            ("华为", "HUAWEI", ["FreeBuds Pro 3", "FreeClip", "FreeBuds SE 2"]),
            ("索尼", "Sony", ["WH-1000XM5", "WF-1000XM5", "WH-CH720N"]),
            ("小米", "Xiaomi", ["Buds 4 Pro", "圈铁耳机 Pro", "Redmi Buds 5"]),
            ("漫步者", "Edifier", ["NeoBuds Pro 2", "W820NB", "花再 Halo"]),
            ("Bose", "Bose", ["QC 45", "大鲨 II", "SoundLink Mini"]),
            ("JBL", "JBL", ["Flip 6", "Tune 520BT", "GO 4"]),
            ("森海塞尔", "Sennheiser", ["Momentum 4", "CX Plus"]),
            ("Beats", "Beats", ["Studio Pro", "Fit Pro"]),
            ("拜亚动力", "Beyerdynamic", ["DT 770 PRO", "Free Byrd"]),
        ],
        "variants": [
            ("标准版", "Standard"), ("降噪版", "ANC"), ("无线版", "Wireless"),
            ("有线版", "Wired"), ("礼盒装", "Gift Box"), ("Pro版", "Pro"),
        ],
    },
    "摄影摄像": {
        "prefix": "CM",
        "price": (1999, 29999),
        "months": _months("back_to_school"),
        "brands": [
            ("佳能", "Canon", ["EOS R50", "EOS R8", "EOS 200D II"]),
            ("索尼", "Sony", ["Alpha 7C II", "ZV-E10", "Alpha 6700"]),
            ("尼康", "Nikon", ["Z 30", "Z 50", "Z fc"]),
            ("富士", "Fujifilm", ["X-S20", "X-T30 II", "instax mini 12"]),
            ("大疆", "DJI", ["Osmo Pocket 3", "Osmo Action 4", "Mini 4 Pro"]),
            ("GoPro", "GoPro", ["HERO12 Black", "HERO11"]),
            ("松下", "Panasonic", ["Lumix G100", "Lumix S5 II"]),
        ],
        "variants": [
            ("单机身", "Body Only"), ("套机 18-55", "Kit 18-55"),
            ("套机 24-70", "Kit 24-70"), ("大师套装", "Pro Kit"),
            ("官方标配", "Standard Kit"), ("旅行套装", "Travel Kit"),
        ],
    },
    "游戏设备": {
        "prefix": "GM",
        "price": (149, 5999),
        "months": _months("back_to_school"),
        "brands": [
            ("任天堂", "Nintendo", ["Switch OLED", "Switch Lite", "Switch 续航版"]),
            ("索尼", "Sony", ["PS5 光驱版", "PS5 数字版", "DualSense 手柄"]),
            ("微软", "Microsoft", ["Xbox Series X", "Xbox Series S", "Xbox 手柄"]),
            ("雷蛇", "Razer", ["炼狱蝰蛇 V3", "黑寡妇 V4", "北海巨妖 V3"]),
            ("罗技", "Logitech", ["G502 X", "G Pro X", "K845 键盘"]),
            ("北通", "BETOP", ["阿修罗 2 Pro", "宙斯 2"]),
            ("八位堂", "8BitDo", ["Pro 2 手柄", "猎户座"]),
        ],
        "variants": [
            ("国行版", "China Version"), ("日版", "Japan Version"), ("港版", "HK Version"),
            ("套装", "Bundle"), ("加强版", "Enhanced"), ("限定版", "Limited Edition"),
        ],
    },
    "数码配件": {
        "prefix": "DG",
        "price": (19, 1299),
        "months": _months("back_to_school"),
        "brands": [
            ("安克", "Anker", ["MagGo", "Nano II", "PowerCore"]),
            ("绿联", "UGREEN", ["闪充套装", "雷电扩展坞", "便携支架"]),
            ("罗马仕", "ROMOSS", ["sense6", "PLUS 快充", "LT20"]),
            ("倍思", "Baseus", ["氮化镓", "磁吸", "灵傲"]),
            ("品胜", "PISEN", ["快充", "数显", "自带线"]),
            ("小米", "Xiaomi", ["移动电源 3", "氮化镓充电器", "USB-C 数据线"]),
            ("贝尔金", "Belkin", ["BoostCharge", "Connect 扩展坞"]),
            ("紫米", "ZMI", ["No.20", "QB826"]),
        ],
        "nouns": [
            ("充电宝", "Power Bank"), ("数据线", "Data Cable"), ("充电器", "Charger"),
            ("扩展坞", "Docking Station"), ("无线鼠标", "Wireless Mouse"), ("键盘", "Keyboard"),
            ("蓝牙音箱", "Bluetooth Speaker"), ("手机支架", "Phone Holder"),
            ("钢化膜", "Screen Protector"), ("保护壳", "Protective Case"),
            ("读卡器", "Card Reader"), ("自拍杆", "Selfie Stick"), ("集线器", "USB Hub"),
            ("车载支架", "Car Mount"),
        ],
        "variants": [
            ("标准版", "Standard"), ("Pro版", "Pro"), ("轻享版", "Lite"),
            ("旗舰版", "Flagship"), ("套装", "Set"),
        ],
    },
    "智能设备": {
        "prefix": "SM",
        "price": (199, 8999),
        "months": _months("back_to_school"),
        "brands": [
            ("小米", "Xiaomi", ["小爱音箱 Pro", "米家扫地机器人", "米家投影仪"]),
            ("华为", "HUAWEI", ["智能音箱 Joy", "AI 音箱 2", "家用摄像头"]),
            ("苹果", "Apple", ["HomePod mini", "Apple TV 4K"]),
            ("石头", "Roborock", ["G20", "Q7 Max"]),
            ("云鲸", "NARWAL", ["J4", "逍遥 001"]),
            ("科大讯飞", "iFLYTEK", ["智能学习机 T20", "翻译笔 S11"]),
            ("大疆", "DJI", ["Osmo Mobile 6", "Ronin 4D"]),
            ("普联", "TP-LINK", ["智能门锁 SL41", "全屋 Wi-Fi 7"]),
        ],
        "nouns": [
            ("智能音箱", "Smart Speaker"), ("扫地机器人", "Robot Vacuum"),
            ("智能门锁", "Smart Lock"), ("家用摄像头", "Security Camera"),
            ("投影仪", "Projector"), ("电子书阅读器", "E-reader"),
            ("空气净化器", "Air Purifier"), ("智能插座", "Smart Plug"),
            ("体脂秤", "Body Scale"), ("智能灯带", "Smart Light Strip"),
            ("学习机", "Study Tablet"), ("智能门铃", "Video Doorbell"),
        ],
        "variants": [
            ("入门版", "Entry"), ("标准版", "Standard"), ("Pro版", "Pro"),
            ("旗舰版", "Flagship"), ("家装套装", "Home Kit"),
        ],
    },
    # ==================== 校园 / 办公（原本很薄的类目） ====================
    "文具用品": {
        "prefix": "ST",
        "price": (2, 299),
        "months": _months("campus"),
        "brands": [
            ("晨光", "M&G", ["优品", "本味", "孔庙祈福"]),
            ("得力", "Deli", ["纽赛", "睿智", "经典"]),
            ("齐心", "Comix", ["优品", "尚品"]),
            ("真彩", "TrueColor", ["经典", "优品"]),
            ("斑马", "Zebra", ["Mildliner", "Sarasa"]),
            ("百乐", "PILOT", ["Juice", "V5"]),
            ("无印良品", "MUJI", ["磨砂", "按动"]),
            ("国誉", "KOKUYO", ["Campus", "淡彩"]),
        ],
        "nouns": [
            ("中性笔", "Gel Pen"), ("按动笔", "Retractable Pen"), ("笔袋", "Pencil Case"),
            ("笔记本", "Notebook"), ("活页本", "Loose-leaf Notebook"), ("荧光笔", "Highlighter"),
            ("2B铅笔", "2B Pencil"), ("橡皮", "Eraser"), ("修正带", "Correction Tape"),
            ("尺子套装", "Ruler Set"), ("订书机", "Stapler"), ("剪刀", "Scissors"),
            ("便利贴", "Sticky Notes"), ("文件夹", "Folder"), ("试卷袋", "Exam Bag"),
            ("计算器", "Calculator"), ("绘图工具", "Drawing Tools"), ("订书钉", "Staples"),
        ],
        "variants": [
            ("单支装", "Single"), ("10支装", "10-Pack"), ("套装", "Set"),
            ("大号", "Large"), ("小号", "Small"), ("学生款", "Student"),
        ],
    },
    "办公用品": {
        "prefix": "OF",
        "price": (3, 3999),
        "months": _months("campus"),
        "brands": [
            ("得力", "Deli", ["经典", "商务"]),
            ("齐心", "Comix", ["优品", "耐用"]),
            ("惠普", "HP", ["LaserJet", "OfficeJet"]),
            ("爱普生", "Epson", ["L3153", "L4268"]),
            ("佳能", "Canon", ["imageCLASS", "PIXMA"]),
            ("3M", "3M", ["Post-it", "Scotch"]),
            ("晨光", "M&G", ["办公", "商务"]),
            ("联想", "Lenovo", ["小新打印", "激光一体机"]),
        ],
        "nouns": [
            ("A4打印纸", "A4 Paper"), ("硒鼓", "Toner Cartridge"), ("墨盒", "Ink Cartridge"),
            ("打印机", "Printer"), ("档案盒", "Archive Box"), ("文件架", "File Rack"),
            ("白板", "Whiteboard"), ("白板笔", "Whiteboard Marker"),
            ("标签打印机", "Label Printer"), ("碎纸机", "Paper Shredder"),
            ("装订机", "Binding Machine"), ("桌面收纳", "Desk Organizer"),
            ("计算器", "Calculator"), ("订书机", "Stapler"), ("文件柜", "File Cabinet"),
            ("胶装机", "Glue Binding Machine"),
        ],
        "variants": [
            ("标准版", "Standard"), ("加大版", "Large"), ("500张装", "500 Sheets"),
            ("1盒装", "1 Box"), ("套装", "Set"), ("商务款", "Business"),
        ],
    },
    "图书音像": {
        "prefix": "BK",
        "price": (12, 399),
        "months": _months("campus"),
        "nouns": [
            ("绘本", "Picture Book"), ("小说", "Novel"), ("词典", "Dictionary"),
            ("字帖", "Calligraphy Copybook"), ("手账本", "Journal"), ("地图册", "Atlas"),
            ("乐谱", "Sheet Music"), ("科普读物", "Science Reader"), ("教辅书", "Study Guide"),
            ("教材", "Textbook"), ("考研真题", "Postgraduate Exam Papers"),
            ("四六级词汇书", "CET Vocabulary Book"), ("专业课讲义", "Course Notes"),
            ("有声书", "Audiobook"), ("拼图", "Jigsaw Puzzle"), ("画册", "Art Book"),
        ],
        "themes": [
            ("大学", "College"), ("精装", "Hardcover"), ("双语", "Bilingual"),
            ("插图版", "Illustrated"), ("典藏", "Collector"), ("新版", "New Edition"),
        ],
        "variants": [
            ("单册", "Single"), ("套装3册", "3-Book Set"), ("套装5册", "5-Book Set"),
            ("全集", "Complete Set"), ("习题册", "Workbook"),
        ],
    },
    "宿舍用品": {
        "prefix": "DM",
        "price": (9, 899),
        "months": _months("campus"),
        "brands": [
            ("网易严选", "YANXUAN", ["简约", "抗菌", "轻奢"]),
            ("小米", "Xiaomi", ["米家", "有品", "极简"]),
            ("南极人", "NANJIREN", ["全棉", "加厚"]),
            ("富安娜", "FUANNA", ["馨而乐", "圣之花"]),
            ("水星家纺", "MERCURY", ["简爱", "寐"]),
            ("宜家", "IKEA", ["简约", "基础款"]),
            ("名创优品", "MINISO", ["简约", "便携"]),
        ],
        "nouns": [
            ("床帘", "Bed Curtain"), ("床上三件套", "Bedding Set"), ("枕头", "Pillow"),
            ("蚊帐", "Mosquito Net"), ("折叠桌", "Folding Desk"), ("台灯", "Desk Lamp"),
            ("电煮锅", "Mini Cooker"), ("插线板", "Power Strip"), ("收纳箱", "Storage Box"),
            ("晾衣架", "Drying Rack"), ("鞋架", "Shoe Rack"), ("小风扇", "Mini Fan"),
            ("加湿器", "Humidifier"), ("洗脸盆套装", "Wash Basin Set"), ("挂钩", "Hooks"),
            ("衣架", "Hangers"), ("保温杯", "Thermos Cup"), ("靠垫", "Cushion"),
        ],
        "variants": [
            ("单人款", "Single"), ("宿舍款", "Dorm"), ("加厚款", "Thickened"),
            ("便携款", "Portable"), ("加大号", "XL"), ("套装", "Set"),
        ],
    },
    # ==================== 其余补充类目 ====================
    "家用电器": {
        "prefix": "HA",
        "price": (99, 6999),
        "months": _months("appliance"),
        "brands": [
            ("美的", "Midea", ["智能", "变频", "家用"]),
            ("九阳", "Joyoung", ["静音", "便携"]),
            ("苏泊尔", "SUPOR", ["大容量", "尊享"]),
            ("戴森", "Dyson", ["无绳", "除菌"]),
            ("飞利浦", "Philips", ["HQ 系列", "经典"]),
            ("松下", "Panasonic", ["纳米水离子", "静音"]),
            ("海尔", "Haier", ["智家", "节能"]),
            ("小米", "Xiaomi", ["米家", "智能"]),
        ],
        "nouns": [
            ("电饭煲", "Rice Cooker"), ("空气炸锅", "Air Fryer"), ("破壁机", "Blender"),
            ("电水壶", "Electric Kettle"), ("吸尘器", "Vacuum Cleaner"), ("加湿器", "Humidifier"),
            ("电风扇", "Electric Fan"), ("取暖器", "Space Heater"), ("电磁炉", "Induction Cooker"),
            ("咖啡机", "Coffee Machine"), ("烤箱", "Oven"), ("洗碗机", "Dishwasher"),
        ],
        "variants": [
            ("基础款", "Basic"), ("标准款", "Standard"), ("尊享款", "Premium"),
            ("旗舰款", "Flagship"), ("套装", "Set"),
        ],
    },
    "运动户外": {
        "prefix": "SP",
        "price": (39, 1899),
        "months": _months("outdoor"),
        "brands": [
            ("迪卡侬", "Decathlon", ["MH500", "QUECHUA", "KIPSTA"]),
            ("探路者", "TOREAD", ["极地", "轻量"]),
            ("李宁", "LI-NING", ["专业", "轻量"]),
            ("安踏", "ANTA", ["竞速", "防泼水"]),
            ("凯乐石", "KAILAS", ["攀岩", "轻量"]),
        ],
        "nouns": [
            ("登山鞋", "Hiking Shoes"), ("冲锋衣", "Outdoor Jacket"), ("帐篷", "Tent"),
            ("睡袋", "Sleeping Bag"), ("登山杖", "Trekking Pole"), ("瑜伽垫", "Yoga Mat"),
            ("哑铃", "Dumbbell"), ("跳绳", "Jump Rope"), ("骑行头盔", "Bike Helmet"),
            ("保温水壶", "Thermos Bottle"), ("运动背包", "Sports Backpack"), ("护膝", "Knee Support"),
        ],
        "variants": [
            ("男款", "Men"), ("女款", "Women"), ("通用款", "Unisex"),
            ("儿童款", "Kids"), ("专业款", "Pro"),
        ],
    },
    "美妆护肤": {
        "prefix": "BT",
        "price": (29, 1299),
        "months": _months("gift_beauty"),
        "brands": [
            ("兰蔻", "Lancome", ["小黑瓶", "轻润"]),
            ("雅诗兰黛", "Estee Lauder", ["小棕瓶", "DW"]),
            ("欧莱雅", "L'Oreal", ["紫熨斗", "小蜜罐"]),
            ("珀莱雅", "PROYA", ["红宝石", "双抗"]),
            ("薇诺娜", "WINONA", ["舒敏", "清透"]),
            ("完美日记", "Perfect Diary", ["名片", "小细跟"]),
            ("花西子", "Florasis", ["同心锁", "雕花"]),
        ],
        "nouns": [
            ("精华液", "Serum"), ("面霜", "Face Cream"), ("洁面乳", "Cleanser"),
            ("防晒霜", "Sunscreen"), ("口红", "Lipstick"), ("粉底液", "Foundation"),
            ("面膜", "Face Mask"), ("眼霜", "Eye Cream"), ("身体乳", "Body Lotion"),
            ("洗发水", "Shampoo"), ("香水", "Perfume"), ("化妆刷", "Makeup Brush"),
        ],
        "variants": [
            ("30ml", "30ml"), ("50ml", "50ml"), ("100ml", "100ml"),
            ("旅行装", "Travel Size"), ("礼盒装", "Gift Box"),
        ],
    },
    "母婴用品": {
        "prefix": "BB",
        "price": (19, 1599),
        "months": _months("family"),
        "brands": [
            ("好孩子", "Goodbaby", ["轻便", "高景观"]),
            ("贝亲", "Pigeon", ["宽口径", "自然实感"]),
            ("帮宝适", "Pampers", ["一级帮", "超薄干爽"]),
            ("花王", "Merries", ["妙而舒", "轻柔"]),
            ("爱他美", "Aptamil", ["卓萃", "经典"]),
            ("babycare", "babycare", ["皇室", "轻氧"]),
        ],
        "nouns": [
            ("婴儿推车", "Baby Stroller"), ("学步鞋", "Toddler Shoes"), ("奶瓶", "Baby Bottle"),
            ("纸尿裤", "Diapers"), ("婴儿床", "Baby Cot"), ("辅食机", "Baby Food Maker"),
            ("浴盆", "Baby Bath Tub"), ("安全座椅", "Car Seat"), ("背带", "Baby Carrier"),
            ("摇铃", "Rattle Toy"), ("隔尿垫", "Changing Pad"), ("口水巾", "Bib"),
        ],
        "variants": [
            ("S码", "Size S"), ("M码", "Size M"), ("L码", "Size L"),
            ("通用款", "Unisex"), ("套装", "Set"),
        ],
    },
    "汽车用品": {
        "prefix": "CA",
        "price": (29, 2999),
        "months": _months("travel"),
        "brands": [
            ("70迈", "70mai", ["A800S", "M500"]),
            ("盯盯拍", "DDPAI", ["MINI5", "Z50"]),
            ("米其林", "Michelin", ["车载", "通用"]),
            ("3M", "3M", ["车蜡", "贴膜"]),
            ("倍思", "Baseus", ["车载", "快充"]),
        ],
        "nouns": [
            ("行车记录仪", "Dash Camera"), ("车载吸尘器", "Car Vacuum"), ("脚垫", "Car Mat"),
            ("座套", "Seat Cover"), ("车载冰箱", "Car Fridge"),
            ("胎压监测", "Tire Pressure Monitor"), ("车蜡", "Car Wax"),
            ("遮阳挡", "Sun Shade"), ("车载香薰", "Car Diffuser"),
            ("应急电源", "Jump Starter"), ("拖车绳", "Tow Rope"), ("车衣", "Car Cover"),
        ],
        "variants": [
            ("通用", "Universal"), ("专用", "Custom Fit"), ("无线", "Wireless"),
            ("便携", "Portable"), ("套装", "Set"),
        ],
    },
    "食品饮料": {
        "prefix": "FD",
        "price": (9, 499),
        "months": _months("food"),
        "brands": [
            ("三只松鼠", "Three Squirrels", ["每日坚果", "精选"]),
            ("良品铺子", "BESTORE", ["高端零食", "精选"]),
            ("雀巢", "Nestle", ["金牌", "醇品"]),
            ("星巴克", "Starbucks", ["星选", "胶囊"]),
            ("农夫山泉", "Nongfu Spring", ["天然水", "东方树叶"]),
            ("伊利", "Yili", ["安慕希", "经典"]),
        ],
        "nouns": [
            ("咖啡豆", "Coffee Beans"), ("茶叶", "Tea Leaves"), ("蜂蜜", "Honey"),
            ("坚果", "Mixed Nuts"), ("巧克力", "Chocolate"), ("饼干", "Biscuits"),
            ("果酱", "Jam"), ("橄榄油", "Olive Oil"), ("意大利面", "Pasta"),
            ("麦片", "Cereal"), ("花茶", "Herbal Tea"), ("调味料", "Seasoning"),
        ],
        "variants": [
            ("250g", "250g"), ("500g", "500g"), ("1kg", "1kg"),
            ("家庭装", "Family Pack"), ("礼盒装", "Gift Box"),
        ],
    },
    "珠宝首饰": {
        "prefix": "JW",
        "price": (89, 12999),
        "months": _months("gift_beauty"),
        "brands": [
            ("周大福", "CHOW TAI FOOK", ["经典", "传承"]),
            ("周生生", "CHOW SANG SANG", ["唯爱", "Pt 系列"]),
            ("潘多拉", "Pandora", ["串饰", "经典"]),
            ("施华洛世奇", "Swarovski", ["白天鹅", "闪耀"]),
            ("APM", "APM Monaco", ["星星", "流星"]),
        ],
        "nouns": [
            ("项链", "Necklace"), ("手链", "Bracelet"), ("戒指", "Ring"),
            ("耳环", "Earrings"), ("吊坠", "Pendant"), ("手镯", "Bangle"),
            ("胸针", "Brooch"), ("脚链", "Anklet"), ("锁骨链", "Collar Chain"),
            ("珍珠耳钉", "Pearl Studs"), ("情侣对戒", "Couple Rings"),
            ("生肖挂件", "Zodiac Charm"),
        ],
        "variants": [
            ("均码", "One Size"), ("可调节", "Adjustable"), ("礼盒装", "Gift Boxed"),
            ("18K金款", "18K Gold"), ("纯银款", "Sterling Silver"),
        ],
    },
    "鞋靴箱包": {
        "prefix": "SH",
        "price": (49, 2999),
        "months": _months("fashion"),
        "brands": [
            ("耐克", "Nike", ["Air Force 1", "Air Max", "Zoom"]),
            ("阿迪达斯", "adidas", ["Ultraboost", "Superstar", "三叶草"]),
            ("李宁", "LI-NING", ["赤兔", "吾适", "音速"]),
            ("安踏", "ANTA", ["氢跑", "马赫", "旅行"]),
            ("新秀丽", "Samsonite", ["商务", "轻量"]),
            ("回力", "Warrior", ["经典", "帆布"]),
        ],
        "nouns": [
            ("运动鞋", "Sneakers"), ("帆布鞋", "Canvas Shoes"), ("短靴", "Ankle Boots"),
            ("长靴", "Long Boots"), ("拖鞋", "Slippers"), ("双肩包", "Backpack"),
            ("单肩包", "Shoulder Bag"), ("手拿包", "Clutch"), ("行李箱", "Suitcase"),
            ("钱包", "Wallet"), ("腰包", "Waist Bag"), ("电脑包", "Laptop Bag"),
        ],
        "variants": [
            ("小号", "Small"), ("中号", "Medium"), ("大号", "Large"),
            ("均码", "One Size"), ("学生款", "Student"),
        ],
    },
    "乐器音响": {
        "prefix": "MS",
        "price": (99, 12999),
        "months": _months("back_to_school"),
        "brands": [
            ("雅马哈", "Yamaha", ["F310", "P-45", "MG10XU"]),
            ("卡西欧", "Casio", ["CT-S1", "PX-S1100"]),
            ("罗兰", "Roland", ["FP-10", "TD-02K"]),
            ("科音", "KORG", ["B2", "microKORG"]),
            ("芬达", "Fender", ["CD-60S", "Player Telecaster"]),
            ("森海塞尔", "Sennheiser", ["e 835", "HD 400 PRO"]),
        ],
        "nouns": [
            ("尤克里里", "Ukulele"), ("吉他", "Guitar"), ("电子琴", "Keyboard Piano"),
            ("口琴", "Harmonica"), ("非洲鼓", "Djembe"), ("麦克风", "Microphone"),
            ("蓝牙音箱", "Bluetooth Speaker"), ("声卡", "Audio Interface"),
            ("调音器", "Tuner"), ("节拍器", "Metronome"), ("效果器", "Effects Pedal"),
            ("监听耳机", "Monitor Headphones"),
        ],
        "variants": [
            ("入门款", "Beginner"), ("专业款", "Professional"), ("便携款", "Portable"),
            ("标准款", "Standard"), ("套装", "Set"),
        ],
    },
}

# 跨类目关联：同一个购物篮里偶尔混入相关类目的商品，让向量学到跨类目关系
# （"买手机的人常顺手买手机壳""买文具的人会买教辅"—— 这些共现正是 Item2Vec 要学的东西）
RELATED: dict[str, list[str]] = {
    "手机": ["数码配件", "耳机音响"],
    "笔记本电脑": ["数码配件", "办公用品"],
    "平板电脑": ["数码配件", "文具用品"],
    "智能穿戴": ["数码配件", "运动户外"],
    "耳机音响": ["数码配件"],
    "摄影摄像": ["数码配件"],
    "游戏设备": ["数码配件", "耳机音响"],
    "数码配件": ["智能设备", "手机"],
    "智能设备": ["数码配件", "家用电器"],
    "家用电器": ["智能设备", "宿舍用品"],
    "文具用品": ["图书音像", "宿舍用品"],
    "办公用品": ["文具用品", "笔记本电脑"],
    "图书音像": ["文具用品"],
    "宿舍用品": ["文具用品", "家用电器"],
    "运动户外": ["鞋靴箱包"],
    "鞋靴箱包": ["运动户外"],
    "美妆护肤": ["珠宝首饰"],
    "珠宝首饰": ["美妆护肤"],
    "家居装饰": ["装饰品", "收纳用品"],
    "装饰品": ["家居装饰", "派对用品"],
    "生活用品": ["厨房用品", "收纳用品"],
    "厨房用品": ["生活用品", "家用电器"],
    "收纳用品": ["生活用品", "宿舍用品"],
    "派对用品": ["装饰品", "食品饮料"],
    "手工用品": ["文具用品"],
    "饰品": ["珠宝首饰"],
    "配饰": ["鞋靴箱包"],
    "家具": ["家居装饰"],
    "洗浴用品": ["美妆护肤"],
    "浴室用品": ["生活用品"],
    "玩具": ["母婴用品"],
    "宠物用品": ["生活用品"],
    "婴儿用品": ["母婴用品"],
    "园艺用品": ["花园用品", "运动户外"],
    "花园用品": ["园艺用品"],
    "工具": ["汽车用品"],
    "购物袋": ["派对用品"],
}


# ------------------------------------------------- 原始类目的"高端补充商品"
# 为什么需要这一层：补充数据 v1/v2 的类目与原始数据**刻意不重叠**，结果是
# "原始客户的真实品类偏好"（家居装饰 / 装饰品 / 厨房用品…）在补充数据里根本没有对应商品，
# 偏好只能退化成随机分配 —— 既不符合真实用户行为，也让"沿用真实偏好"这句话落不了地。
#
# 所以这里用**原始商品的真实名称**派生一批"同品类高端款"：
#   原始：白色悬挂心形茶烛台（1.29 元）
#   补充：轻奢 白色悬挂心形茶烛台 大号（约 300 元）
# 于是补充数据覆盖的类目与原始数据完全一致，价格带也被打开（高价家居装饰终于存在了），
# 而商品编码用独立前缀（HM / DC / LX…），一眼仍能区分数据来源。
RAW_FILLER_SPECS: dict[str, dict[str, Any]] = {
    "家居装饰": {"prefix": "HM", "price": (39, 2999), "months": _months("home_decor")},
    "装饰品": {"prefix": "DC", "price": (19, 1299), "months": _months("festive")},
    "生活用品": {"prefix": "LX", "price": (19, 899), "months": _months("home_decor")},
    "厨房用品": {"prefix": "KC", "price": (29, 2499), "months": _months("home_decor")},
    "派对用品": {"prefix": "PD", "price": (19, 899), "months": _months("festive")},
    "收纳用品": {"prefix": "SN", "price": (29, 1299), "months": _months("home_decor")},
    "手工用品": {"prefix": "HD", "price": (19, 999), "months": _months("festive")},
    "购物袋": {"prefix": "BD", "price": (9, 399), "months": {}},
    "饰品": {"prefix": "JS", "price": (39, 1999), "months": _months("gift_beauty")},
    "园艺用品": {"prefix": "YY", "price": (29, 999), "months": _months("gardening")},
    "花园用品": {"prefix": "HY", "price": (39, 1999), "months": _months("gardening")},
    "玩具": {"prefix": "WJ", "price": (29, 899), "months": _months("festive")},
    "宠物用品": {"prefix": "CW", "price": (19, 799), "months": {}},
    "工具": {"prefix": "GJ", "price": (39, 1999), "months": _months("tools")},
    "家具": {"prefix": "JJ", "price": (199, 6999), "months": _months("home_decor")},
    "配饰": {"prefix": "PS", "price": (29, 1299), "months": _months("gift_beauty")},
    "洗浴用品": {"prefix": "XY", "price": (19, 699), "months": _months("summer_bath")},
    "浴室用品": {"prefix": "YS", "price": (29, 1299), "months": _months("summer_bath")},
    "婴儿用品": {"prefix": "YE", "price": (29, 999), "months": _months("family")},
}

# 高端款的名字 = 主题词 + 原始商品名 + 规格词
FILLER_THEMES: list[tuple[str, str]] = [
    ("臻选", "Premium"), ("典藏", "Collector"), ("北欧", "Nordic"),
    ("轻奢", "Luxury"), ("手工", "Handmade"), ("复古", "Vintage"),
    ("设计师", "Designer"),
]
FILLER_VARIANTS: list[tuple[str, str]] = [
    ("大号", "Large"), ("套装", "Set"), ("限定版", "Limited Edition"),
    ("礼盒装", "Gift Box"), ("加厚款", "Thickened"),
]

# 脏数据注入比例（与原始数据的实际比例接近，保证清洗漏斗真的在干活）
MISSING_USER_RATE = 0.085
RETURN_RATE = 0.025
BAD_PRICE_RATE = 0.008

# 订单落在"客户偏好类目"里的比例（其余是尝鲜/跨类目购买）
PREFERENCE_RATE = 0.72


# ---------------------------------------------------------------- 人物画像
# 用"剧本"生成整年连贯的购买记录：每个 trip 是一次采购（一天一张发票），
# 里面写 1~3 个品类。画像的交易**不参与脏数据注入**（演示用，必须干净可用）。
PERSONAS: list[dict[str, Any]] = [
    {
        "customer_id": "QZJ001",
        "display_name": "QZJ",
        "label": "大学生 QZJ（数码 / 文具 / 图书 / 宿舍用品，开学季集中采购）",
        "country": "United Kingdom",
        # (日期, [品类…], 剧情说明)
        "trips": [
            ("2010-12-05", ["宿舍用品", "文具用品"], "入学安顿：床帘、台灯、文具"),
            ("2010-12-20", ["数码配件"], "数据线与充电器"),
            ("2011-01-18", ["笔记本电脑", "数码配件"], "换笔记本 + 鼠标"),
            ("2011-02-22", ["耳机音响"], "开学后买耳机"),
            ("2011-03-10", ["图书音像", "文具用品"], "教材与笔记本"),
            ("2011-04-16", ["运动户外"], "体育课装备"),
            ("2011-05-12", ["图书音像"], "期中考试资料"),
            ("2011-06-18", ["手机", "数码配件"], "618 换手机 + 手机壳 / 钢化膜"),
            ("2011-07-06", ["宿舍用品", "食品饮料"], "暑期留校：小风扇、零食"),
            ("2011-08-24", ["平板电脑", "智能穿戴", "宿舍用品"], "开学季采购：平板 + 手环 + 宿舍用品"),
            ("2011-09-03", ["文具用品", "图书音像"], "开学文具与教辅"),
            ("2011-09-12", ["数码配件"], "扩展坞、U 盘、键盘"),
            ("2011-10-08", ["鞋靴箱包"], "秋季换季：运动鞋与双肩包"),
            ("2011-11-11", ["耳机音响", "智能穿戴", "家用电器"], "双 11：降噪耳机 + 手环 + 加湿器"),
            ("2011-11-25", ["图书音像"], "考研资料"),
            ("2011-12-06", ["食品饮料", "数码配件"], "期末周：零食与充电宝"),
        ],
    },
]


def _combos(spec: Mapping[str, Any]) -> list[tuple[str, str]]:
    """把类目定义展开成 ``[(中文名, 英文名), …]``。

    三种组合方式：

    * ``brands`` + ``nouns``（家电、文具、鞋包…）→「品牌 + 名词 + 规格」，
      例如 `小米 空气炸锅 尊享款`；
    * 只有 ``brands``（手机、笔记本这些"型号即卖点"的 3C）→「品牌 + 型号 + 规格」，
      例如 `苹果 iPhone 15 Pro 256GB`；
    * 两者都没有 →「系列词 + 名词 + 规格词」，例如 `磁吸充电器 Pro版`。
    """
    variants = spec["variants"]
    brands = spec.get("brands") or []
    nouns = spec.get("nouns") or []
    if brands and nouns:
        return [
            (f"{b_cn} {n_cn} {v_cn}".strip(), f"{b_en} {n_en} {v_en}".strip())
            for b_cn, b_en, _series in brands
            for n_cn, n_en in nouns
            for v_cn, v_en in variants
        ]
    if brands:
        out: list[tuple[str, str]] = []
        for brand_cn, brand_en, models in brands:
            for model in models:
                for var_cn, var_en in variants:
                    out.append(
                        (
                            f"{brand_cn} {model} {var_cn}".strip(),
                            f"{brand_en} {model} {var_en}".strip(),
                        )
                    )
        return out
    return [
        (f"{t_cn}{n_cn} {v_cn}".strip(), f"{t_en} {n_en} {v_en}".strip())
        for n_cn, n_en in spec["nouns"]
        for t_cn, t_en in spec.get("themes", [("", "")])
        for v_cn, v_en in variants
    ]


def _build_products(rng: np.random.Generator, per_category: int) -> pd.DataFrame:
    """按类目生成商品主数据（StockCode / 英文名 / 中文名 / 类目）。"""
    rows: list[dict[str, str]] = []
    for category, spec in CATEGORY_SPECS.items():
        combos = _combos(spec)
        count = min(per_category, len(combos))
        picked = sorted(int(p) for p in rng.choice(len(combos), size=count, replace=False))
        for idx, pos in enumerate(picked, start=1):
            cn_name, en_name = combos[pos]
            rows.append(
                {
                    "StockCode": f"{spec['prefix']}{10000 + idx}",
                    "Description": en_name,
                    "Chinese_Description": cn_name,
                    "Product_Category": category,
                }
            )
    return pd.DataFrame(rows)


def _build_filler_products(
    rng: np.random.Generator,
    raw_products: pd.DataFrame,
    per_category: int,
) -> pd.DataFrame:
    """用**原始商品名**派生"同品类高端款"，把原始 19 个类目也覆盖上。

    这样"客户的真实品类偏好"才落得到具体商品上（见 :data:`RAW_FILLER_SPECS` 的说明）。
    """
    rows: list[dict[str, str]] = []
    for category, spec in RAW_FILLER_SPECS.items():
        source = raw_products[raw_products["Product_Category"].astype(str) == category]
        if source.empty:
            continue
        names = source.dropna(subset=["Chinese_Description"]).drop_duplicates("Chinese_Description")
        if names.empty:
            continue
        positions = sorted(
            int(p) for p in rng.choice(len(names), size=min(per_category, len(names)), replace=False)
        )
        picked = names.iloc[positions]
        for idx, (_, item) in enumerate(picked.iterrows(), start=1):
            theme_cn, theme_en = FILLER_THEMES[int(rng.integers(len(FILLER_THEMES)))]
            var_cn, var_en = FILLER_VARIANTS[int(rng.integers(len(FILLER_VARIANTS)))]
            base_cn = str(item["Chinese_Description"]).strip()
            base_en = str(item.get("Description") or base_cn).strip()
            rows.append(
                {
                    "StockCode": f"{spec['prefix']}{10000 + idx}",
                    "Description": f"{theme_en} {base_en} {var_en}".strip(),
                    "Chinese_Description": f"{theme_cn} {base_cn} {var_cn}".strip(),
                    "Product_Category": category,
                }
            )
    return pd.DataFrame(rows)


def _build_prices(
    rng: np.random.Generator, products: pd.DataFrame, specs: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    """给每个商品定价：类目区间内取**对数均匀**分布（便宜的多、贵的少），再小幅抖动。"""
    prices: dict[str, float] = {}
    for category, group in products.groupby("Product_Category"):
        low, high = specs[str(category)]["price"]
        samples = np.exp(rng.uniform(np.log(low), np.log(high), size=len(group)))
        for code, price in zip(group["StockCode"], samples):
            prices[str(code)] = float(np.round(float(price) * rng.uniform(0.92, 1.08), 2))
    return prices


def _day_sampler(
    date_start: str, date_end: str, specs: Mapping[str, Mapping[str, Any]]
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """按"类目月度权重"抽日期的采样器。

    返回 ``(日期列表, {类目: 概率数组})``：概率里已经带上月度权重，
    所以 8~9 月的订单会自然地更多落在数码/文具类目上，11~12 月的落在礼赠/家电类目上。
    """
    days = pd.date_range(pd.Timestamp(date_start).normalize(), pd.Timestamp(date_end).normalize())
    month_of_day = np.array([d.month for d in days])
    probs: dict[str, np.ndarray] = {}
    for category, spec in specs.items():
        months = spec.get("months") or {}
        weights = np.array([float(months.get(int(m), 1.0)) for m in month_of_day])
        probs[category] = weights / weights.sum()
    return days, probs


def _build_persona_transactions(
    *,
    rng: np.random.Generator,
    pools: Mapping[str, tuple[list[str], np.ndarray]],
    prices: Mapping[str, float],
    desc_by_code: Mapping[str, str],
    personas: Sequence[Mapping[str, Any]],
    start_invoice: int,
    source_label: str,
) -> pd.DataFrame:
    """按剧本生成人物画像的购买记录（一天一张发票，每个品类 1~2 件）。"""
    records: list[dict[str, Any]] = []
    invoice_no = start_invoice
    for persona in personas:
        customer = str(persona["customer_id"])
        country = str(persona.get("country", "United Kingdom"))
        for trip_date, trip_categories, _note in persona["trips"]:
            day = pd.Timestamp(trip_date)
            stamp = day + pd.Timedelta(
                hours=int(rng.integers(10, 21)), minutes=int(rng.integers(0, 60))
            )
            date_text = f"{stamp.month}/{stamp.day}/{stamp.year} {stamp.hour}:{stamp.minute:02d}"
            invoice_no += 1
            for category in trip_categories:
                codes, weights = pools[str(category)]
                n = int(rng.integers(1, 3))  # 每个品类 1~2 件
                for code in rng.choice(codes, size=min(n, len(codes)), replace=False, p=weights):
                    code = str(code)
                    records.append(
                        {
                            "InvoiceNo": str(invoice_no),
                            "StockCode": code,
                            "Description": f"{source_label} {desc_by_code.get(code, code)}".strip(),
                            "Quantity": int(rng.integers(1, 3)),
                            "InvoiceDate": date_text,
                            "UnitPrice": float(prices[code]),
                            "CustomerID": customer,
                            "Country": country,
                        }
                    )
    return pd.DataFrame.from_records(records)


def _build_transactions(
    *,
    rng: np.random.Generator,
    products: pd.DataFrame,
    prices: dict[str, float],
    existing_customers: Sequence[str],
    new_customer_count: int,
    invoice_count: int,
    start_invoice: int,
    country_pool: Sequence[str],
    date_start: str,
    date_end: str,
    source_label: str,
    personas: Sequence[Mapping[str, Any]] = (),
    customer_preferences: Mapping[str, Sequence[str]] | None = None,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """生成交易明细：季节性日期 + 主题购物篮 + 长尾热度 + 脏数据注入 + 人物画像。"""
    desc_by_code = dict(zip(products["StockCode"].astype(str), products["Description"].astype(str)))

    # 每个类目的商品池；权重按 Zipf 递减，制造"少数商品很热"的长尾
    pools: dict[str, tuple[list[str], np.ndarray]] = {}
    for category, group in products.groupby("Product_Category"):
        codes = [str(c) for c in group["StockCode"]]
        weights = 1.0 / np.power(np.arange(1, len(codes) + 1), 0.85)
        pools[str(category)] = (codes, weights / weights.sum())
    categories = list(pools)

    # 客户池：多数沿用已有客户（让新旧数据真正交融），一部分是新客户
    all_customers = list(existing_customers) + [
        f"S{90000 + i}" for i in range(1, new_customer_count + 1)
    ]
    if not all_customers:
        all_customers = [f"S{90000 + i}" for i in range(1, new_customer_count + 1)]
    # 少量"重度客户"，下单频次远高于常人，符合真实长尾
    weights = np.ones(len(all_customers))
    heavy = rng.choice(len(all_customers), size=max(1, len(all_customers) // 60), replace=False)
    weights[heavy] = 8.0
    customer_weights = weights / weights.sum()

    # ---- 客户偏好画像：真实用户有稳定的品类偏好，不是每次都随机逛 ----
    # 有原始购买记录的客户，直接用他在原始数据里最常买的 1~3 个类目；
    # 新客户随机分配 1~3 个偏好类目。没有这一层，用户-商品矩阵就是噪声。
    given = customer_preferences or {}
    preferences: dict[str, list[str]] = {}
    for customer in all_customers:
        pool = [c for c in given.get(str(customer), []) if c in pools]
        if not pool:
            k = int(rng.integers(1, 4))
            pool = [str(c) for c in rng.choice(categories, size=k, replace=False)]
        preferences[str(customer)] = pool

    days, day_probs = _day_sampler(
        date_start, date_end, specs if specs is not None else CATEGORY_SPECS
    )
    countries = list(country_pool) or ["United Kingdom"]

    records: list[dict[str, Any]] = []
    for i in range(invoice_count):
        invoice = str(start_invoice + i)
        customer = all_customers[int(rng.choice(len(all_customers), p=customer_weights))]
        # 七成订单买偏好类目，三成逛别的类目（保留探索性，也让跨类目共现存在）
        pool = preferences[str(customer)]
        if pool and rng.random() < PREFERENCE_RATE:
            category = str(rng.choice(pool))
        else:
            category = str(rng.choice(categories))
        codes, weights_c = pools[category]

        # 一个购物篮 1~8 件商品，主要来自同一类目
        n_items = int(rng.integers(1, 9))
        picked = list(rng.choice(codes, size=min(n_items, len(codes)), replace=False, p=weights_c))

        # 三成购物篮混入 1~2 件相关类目的商品，制造跨类目共现
        related = RELATED.get(category)
        if related and rng.random() < 0.30:
            rel = str(rng.choice(related))
            r_codes, r_weights = pools[rel]
            # 相关类目也可能只有一两个商品（例如原始数据里的"花园用品"），
            # 取样个数必须按池子大小截断，否则 numpy 会直接报错。
            size = min(int(rng.integers(1, 3)), len(r_codes))
            picked += list(rng.choice(r_codes, size=size, replace=False, p=r_weights))

        country = str(rng.choice(countries))
        # 先按月度权重挑一天，再在当天随机挑一个营业时间
        day = days[int(rng.choice(len(days), p=day_probs[category]))]
        stamp = day + pd.Timedelta(
            hours=int(rng.integers(9, 20)), minutes=int(rng.integers(0, 60))
        )
        date_text = f"{stamp.month}/{stamp.day}/{stamp.year} {stamp.hour}:{stamp.minute:02d}"

        for code in picked:
            code = str(code)
            records.append(
                {
                    "InvoiceNo": invoice,
                    "StockCode": code,
                    "Description": f"{source_label} {desc_by_code.get(code, code)}".strip(),
                    "Quantity": int(np.clip(rng.lognormal(0.9, 0.9), 1, 240)),
                    "InvoiceDate": date_text,
                    "UnitPrice": prices[code],
                    "CustomerID": customer,
                    "Country": country,
                }
            )

    df = pd.DataFrame.from_records(records)

    # ---- 注入退货单：独立发票号（以 C 开头）、数量为负 ----
    n_return = max(1, int(len(df) * RETURN_RATE / 3))
    picked_idx = rng.choice(len(df), size=min(n_return, len(df)), replace=False)
    returns = df.iloc[picked_idx].copy()
    returns["InvoiceNo"] = [f"C{start_invoice + invoice_count + j}" for j in range(len(returns))]
    returns["Quantity"] = -returns["Quantity"].abs()
    df = pd.concat([df, returns], ignore_index=True)

    # ---- 注入缺失用户 ID ----
    df.loc[rng.random(len(df)) < MISSING_USER_RATE, "CustomerID"] = ""

    # ---- 注入异常价格 ----
    df.loc[rng.random(len(df)) < BAD_PRICE_RATE, "UnitPrice"] = 0

    # ---- 人物画像：放在脏数据注入之后，保证画像数据干净可用 ----
    persona_rows = _build_persona_transactions(
        rng=rng,
        pools=pools,
        prices=prices,
        desc_by_code=desc_by_code,
        personas=personas,
        start_invoice=start_invoice + invoice_count + 100,
        source_label=source_label,
    )
    if not persona_rows.empty:
        df = pd.concat([df, persona_rows], ignore_index=True)

    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def generate_synthetic(
    *,
    seed: int = 42,
    per_category: int = 200,
    invoice_count: int = 78000,
    existing_customers: Sequence[str] = (),
    new_customer_count: int = 900,
    start_invoice: int = 800001,
    country_pool: Sequence[str] = ("United Kingdom",),
    date_start: str = "2010-12-01",
    date_end: str = "2011-12-09",
    source_label: str = "",
    personas: Sequence[Mapping[str, Any]] = tuple(PERSONAS),
    customer_preferences: Mapping[str, Sequence[str]] | None = None,
    raw_products: pd.DataFrame | None = None,
    filler_per_category: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成 ``(交易表, 商品表)``，两张表的结构与原始数据完全一致。

    ``date_start`` / ``date_end`` 默认与原始数据的时间范围一致
    （2010-12-01 ~ 2011-12-09），这样"所有数据源都覆盖的窗口"就是整段时间，
    季节统计不会因为某个来源缺了某个月而失真。
    """
    rng = np.random.default_rng(seed)
    products = _build_products(rng, per_category)

    # 原始类目的"高端补充商品"：让客户的真实品类偏好能落到具体商品上
    specs: dict[str, Mapping[str, Any]] = dict(CATEGORY_SPECS)
    if raw_products is not None and filler_per_category > 0:
        fillers = _build_filler_products(rng, raw_products, filler_per_category)
        if not fillers.empty:
            products = pd.concat([products, fillers], ignore_index=True)
        specs.update(RAW_FILLER_SPECS)

    prices = _build_prices(rng, products, specs)
    transactions = _build_transactions(
        rng=rng,
        products=products,
        prices=prices,
        existing_customers=existing_customers,
        new_customer_count=new_customer_count,
        invoice_count=invoice_count,
        start_invoice=start_invoice,
        country_pool=country_pool,
        date_start=date_start,
        date_end=date_end,
        source_label=source_label,
        personas=personas,
        customer_preferences=customer_preferences,
        specs=specs,
    )
    return transactions, products
