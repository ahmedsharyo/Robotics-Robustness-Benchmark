from src.utils.metrics import summarize

def test_summary_metrics():
    rows=[{"episode_return":1,"success":True,"endpoint_error":.1,"steps":4,"control_effort":2,"joint_limit_violations":0},{"episode_return":-1,"success":False,"endpoint_error":.3,"steps":8,"control_effort":4,"joint_limit_violations":1}]
    s=summarize(rows); assert s["success_rate"]==.5 and s["mean_endpoint_error"]==.2 and s["joint_limit_violations"]==1
