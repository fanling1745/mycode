# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 基于云南省立体气候的土壤类型预测模拟 (海拔+降水+时间→土壤类型)
# 输入: 海拔(m) / 年降水(mm) / 发育阶段(初期/中期/成熟期)
# 输出: 预测土壤类型 + 成土特征 + 剖面结构描述
# 使用: 修改 sim.simulate() 参数 → 运行
# 依赖: 无
# =============================================================================
class YunnanSoilSimulator:
    """
    云南省立体气候土壤发生模拟器
    """
    def __init__(self):
        # 基础设定：假设海平面基准温度为 25°C，气温递减率约为 0.6°C / 100m
        self.base_temp = 25.0

    def calculate_temperature(self, elevation):
        """根据海拔计算大致年均温"""
        return self.base_temp - (elevation / 100) * 0.6

    def simulate(self, elevation, precipitation, time_stage="成熟期"):
        """
        根据输入参数模拟土壤类型和剖面特征
        :param elevation: 海拔 (米)
        :param precipitation: 年降水量 (毫米)
        :param time_stage: 发育时间 ('初期', '中期', '成熟期')
        """
        temp = self.calculate_temperature(elevation)
        print(f"--- 模拟参数 ---")
        print(f"📍 海拔: {elevation} m | 🌡️ 预估年均温: {temp:.1f}°C | 🌧️ 年降水: {precipitation} mm | ⏳ 阶段: {time_stage}")
        print(f"----------------")

        # 1. 拦截时间因素 (如果是初期或中期，土壤未完全发育)
        if time_stage == "初期":
            self._print_result("岩石风化壳 / 粗骨土",
                               "土壤刚刚开始发育，风化作用为主。",
                               "极薄的A层 (几厘米) -> C层 (碎石母质) -> R层 (基岩)")
            return
        elif time_stage == "中期":
            self._print_result("幼年土",
                               "有机质开始积累，颜色逐渐加深，但淋溶淀积特征不明显。",
                               "较薄的A层 -> 正在发育的Bw层 -> C层 -> R层")
            return

        # 2. 核心逻辑：基于云南垂直气候带的成熟期土壤分类
        if elevation < 800:
            if precipitation > 1200:
                soil_type = "砖红壤 / 砖红壤性红壤"
                desc = "地处云南南部（如西双版纳），热带季风气候，高温高湿。风化淋溶作用极其强烈，盐基离子流失殆尽，铁铝氧化物大量富集，土壤呈鲜红色，呈酸性。"
                profile = "薄O/A层 (有机质分解极快) -> 深厚的红/砖红色B层 (铁铝淀积) -> C层"
            else:
                soil_type = "热带燥红土"
                desc = "地处干热河谷地带（如元江流域），焚风效应导致高温少雨，蒸发强烈。"
                profile = "极薄A层 -> 红褐色B层 (偶有钙积形态) -> C层"

        elif 800 <= elevation < 2500:
            if precipitation >= 1000:
                soil_type = "黄壤"
                desc = "常年云雾缭绕、相对湿度极大的中山地带，氧化铁发生水化作用，土壤呈现黄色。"
                profile = "中等厚度O/A层 -> 黄色B层 -> C层"
            else:
                soil_type = "红壤" # 云南最广泛的土壤
                desc = "滇中高原典型土壤（如昆明、玉溪）。亚热带季风气候，干湿季分明。具有明显的富铝化过程，但不如砖红壤强烈。"
                profile = "较厚的暗色A层 -> 红色B层 (常见网纹斑块) -> C层"

        elif 2500 <= elevation < 3500:
            soil_type = "棕壤 / 暗棕壤"
            desc = "滇西北及高海拔山区，气候冷凉湿润。针阔混交林下有机质分解缓慢，积累较多，淋溶作用中等。"
            profile = "深厚的O层 (枯枝落叶) -> 暗黑色A层 -> 棕/褐色B层 -> C层"

        else: # elevation >= 3500
            soil_type = "高山草甸土 / 寒漠土"
            desc = "极高山地带（如迪庆、丽江高山）。气候严寒，生物化学风化极其微弱，受冻融物理风化影响大。植被为高山草甸，根系密集盘结。"
            profile = "致密的草根垫层(O/A层) -> 砾石含量极高的过渡层 -> R层 (基岩)"

        self._print_result(soil_type, desc, profile)

    def _print_result(self, soil_type, desc, profile):
        print(f"🌱 预测土壤类型: {soil_type}")
        print(f"📝 成土特征: {desc}")
        print(f"🧱 剖面结构: {profile}\n")


# ==========================================
# 运行测试用例
# ==========================================
if __name__ == "__main__":
    simulator = YunnanSoilSimulator()

    # 测试 1: 西双版纳的热带雨林 (低海拔，高降水，成熟期)
    print("【案例 1：滇南热带谷地】")
    simulator.simulate(elevation=600, precipitation=1500, time_stage="成熟期")

    # 测试 2: 昆明附近的滇中高原 (中海拔，中等降水，成熟期)
    print("【案例 2：滇中红土高原】")
    simulator.simulate(elevation=1900, precipitation=900, time_stage="成熟期")

    # 测试 3: 香格里拉的高山地带 (高海拔，冷凉，成熟期)
    print("【案例 3：滇西北高寒山区】")
    simulator.simulate(elevation=3600, precipitation=600, time_stage="成熟期")

    # 测试 4: 元江干热河谷 (低海拔，低降水，初期发育)
    print("【案例 4：新近发生滑坡的干热河谷】")
    simulator.simulate(elevation=500, precipitation=400, time_stage="初期")