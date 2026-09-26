"""Independent arithmetic/sign check of installed Itzi 25.4 exchange methods.

Uses a node test double, not a routed SWMM network. Do not describe this as
the requested full single-inlet/single-conduit validation case.
"""
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from itzi.drainage import DrainageNode
from audit_reviewer_v5_evidence import OUT


def main():
    assert importlib.metadata.version("itzi") == "25.4"
    rows=[]
    for name,head,h,area,dt in [("free_weir",-1.,.1,400.,.5),
                              ("return_orifice",.5,.1,400.,.5),
                              ("dry_cell_limiter",-1.,.1,.001,.5),
                              ("equal_heads",.1,.1,400.,.5)]:
        sink=[]
        node=DrainageNode.__new__(DrainageNode)
        node.g=9.81;node.surface_area=1.;node.weir_width=2*math.sqrt(math.pi)
        node.orifice_coeff=.6;node.free_weir_coeff=.6;node.submerged_weir_coeff=.6
        node.relaxation_factor=1.;node.damping_factor=1.;node.coupling_flow=0.
        node.pyswmm_node=SimpleNamespace(head=head,generated_inflow=sink.append)
        expected=(-2/3*.6*node.weir_width*h**1.5*math.sqrt(2*9.81) if head<0 else
                  .6*math.sqrt(2*9.81*(head-h)) if head>h else 0.)
        expected=max(expected,-h*area/dt)
        node.apply_coupling(0.,h,dt,area)
        assert math.isclose(node.coupling_flow,expected,abs_tol=1e-12)
        assert math.isclose((node.coupling_flow+sink[0])*dt,0.,abs_tol=1e-12)
        rows.append({"case":name,"computed_surface_source_m3s":node.coupling_flow,
                     "independent_formula_m3s":expected,"swmm_generated_inflow_m3s":sink[0],
                     "exchange_volume_sum_m3":(node.coupling_flow+sink[0])*dt,"passed":True})
    source=Path(inspect.getfile(DrainageNode))
    result={"status":"PASS","scope":"Native exchange function with prescribed-head test double; no pipe routing",
            "itzi_version":"25.4","source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"cases":rows}
    path=OUT/"diagnostics/native_exchange_unit_check.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
