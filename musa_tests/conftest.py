import pytest

def _should_skip(item):
    """判断用例是否会被 skip，用于在收集阶段 deselect 以排除出 Allure 报告。"""
    # 无条件 skip
    if item.get_closest_marker("skip"):
        return True
    # 条件 skip：检查所有 skipif marker，任一条件为 True 则 skip
    for marker in item.iter_markers("skipif"):
        if marker.args and marker.args[0]:
            return True
    return False


def pytest_collection_modifyitems(config, items):
    """将会被 skip 的用例 deselect，使其不出现在 Allure 报告中，
    从而不影响 pass rate 统计。"""
    selected = []
    deselected = []
    for item in items:
        if _should_skip(item):
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected