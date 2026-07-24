# Generates 4 self-contained SVG diagrams for the UAV deployment build.
import html

PALETTE = dict(
    ink="#14213d", muted="#5a6b86", line="#8aa0bd",
    power="#e2622c", data="#1f5fbf", rf="#159a6b", timing="#8a4fc4",
    box="#ffffff",
)
DOMAIN = {
    "power":  ("#fbe9df", "#e2622c"),
    "compute":("#e7effb", "#1f5fbf"),
    "flight": ("#eaf7f0", "#159a6b"),
    "nav":    ("#efe9fb", "#8a4fc4"),
    "payload":("#fff6e0", "#d4a017"),
    "radio":  ("#e6f7fb", "#0e8aa8"),
    "sec":    ("#fdeaea", "#c0392b"),
    "cloud":  ("#eef1f6", "#41506b"),
    "edge":   ("#e7effb", "#1f5fbf"),
    "onboard":("#eaf7f0", "#159a6b"),
    "field":  ("#eef7ea", "#5a8f2b"),
}

def esc(s): return html.escape(str(s))

class SVG:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.parts = []
        self.title = title
    def rect(self, x,y,w,h, fill="#fff", stroke="#8aa0bd", rx=8, sw=1.4, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
    def box(self, x,y,w,h, title, lines=None, domain="compute", fs=14, tfs=None):
        fill,stroke = DOMAIN[domain]
        self.rect(x,y,w,h, fill=fill, stroke=stroke, rx=9, sw=1.8)
        self.rect(x,y,w,26, fill=stroke, stroke=stroke, rx=9, sw=1.8)
        self.parts.append(f'<rect x="{x}" y="{y+13}" width="{w}" height="13" fill="{stroke}"/>')
        self.text(x+w/2, y+18, title, fs=(tfs or 13), col="#fff", anchor="middle", bold=True)
        if lines:
            for i,ln in enumerate(lines):
                self.text(x+11, y+44+i*17, ln, fs=fs-1, col=PALETTE["ink"], anchor="start")
    def text(self, x,y,s, fs=13, col="#14213d", anchor="start", bold=False, italic=False, rot=None):
        b = ' font-weight="700"' if bold else ""
        it = ' font-style="italic"' if italic else ""
        tr = f' transform="rotate({rot} {x} {y})"' if rot else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
                          f'font-size="{fs}" fill="{col}" text-anchor="{anchor}"{b}{it}{tr}>{esc(s)}</text>')
    def line(self, x1,y1,x2,y2, kind="data", label=None, dash=None, arrow="end", lx=None, ly=None):
        col = PALETTE[kind]
        d = dash or ("6 4" if kind=="rf" else ("2 4" if kind=="timing" else None))
        da = f' stroke-dasharray="{d}"' if d else ""
        m = f' marker-end="url(#arw-{kind})"' if arrow in ("end","both") else ""
        ms = f' marker-start="url(#arw-{kind})"' if arrow in ("both",) else ""
        sw = 3.2 if kind=="power" else 2.2
        self.parts.append(f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{col}" '
                          f'stroke-width="{sw}"{da}{m}{ms}/>')
        if label:
            mx,my = (lx if lx is not None else (x1+x2)/2), (ly if ly is not None else (y1+y2)/2)
            tw = len(label)*6.0+8
            self.parts.append(f'<rect x="{mx-tw/2}" y="{my-9}" width="{tw}" height="15" rx="3" fill="#ffffff" opacity="0.92"/>')
            self.text(mx, my+2.5, label, fs=10.5, col=col, anchor="middle", bold=True)
    def elbow(self, x1,y1,x2,y2, kind="data", label=None, midx=None):
        col=PALETTE[kind]; mx = midx if midx is not None else (x1+x2)/2
        d = ("6 4" if kind=="rf" else ("2 4" if kind=="timing" else None))
        da=f' stroke-dasharray="{d}"' if d else ""
        sw=3.2 if kind=="power" else 2.2
        self.parts.append(f'<path d="M {x1} {y1} L {mx} {y1} L {mx} {y2} L {x2} {y2}" fill="none" '
                          f'stroke="{col}" stroke-width="{sw}"{da} marker-end="url(#arw-{kind})"/>')
        if label:
            tw=len(label)*6.0+8
            self.parts.append(f'<rect x="{mx-tw/2}" y="{(y1+y2)/2-9}" width="{tw}" height="15" rx="3" fill="#fff" opacity="0.92"/>')
            self.text(mx,(y1+y2)/2+2.5,label,fs=10.5,col=col,anchor="middle",bold=True)
    def legend(self, x, y, items):
        self.rect(x,y,232,24+len(items)*20, fill="#fbfcfe", stroke="#c7d2e0", rx=7)
        self.text(x+12,y+18,"Legend",fs=12,bold=True)
        for i,(kind,lab) in enumerate(items):
            yy=y+40+i*20
            col=PALETTE[kind]; d=("6 4" if kind=="rf" else ("2 4" if kind=="timing" else None))
            da=f' stroke-dasharray="{d}"' if d else ""
            sw=3.2 if kind=="power" else 2.2
            self.parts.append(f'<path d="M {x+12} {yy} L {x+44} {yy}" stroke="{col}" stroke-width="{sw}"{da}/>')
            self.text(x+52,yy+4,lab,fs=11)
    def svg(self):
        defs = ['<defs>']
        for kind in ("data","power","rf","timing"):
            c=PALETTE[kind]
            defs.append(f'<marker id="arw-{kind}" viewBox="0 0 10 10" refX="9" refY="5" '
                        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                        f'<path d="M0 0 L10 5 L0 10 z" fill="{c}"/></marker>')
        defs.append('</defs>')
        head=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
              f'font-family="Arial, Helvetica, sans-serif">')
        bg=f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="#ffffff"/>'
        t=(f'<text x="30" y="38" font-size="22" font-weight="700" fill="#14213d">{esc(self.title)}</text>')
        return "\n".join([head]+defs+[bg,t]+self.parts+["</svg>"])
    def save(self, path):
        open(path,"w").write(self.svg())
        print("wrote", path)

# ============================================================ Diagram 1: wiring
d = SVG(1240, 880, "Diagram 1  —  UAV Onboard Wiring / Integration")
d.text(30,60,"Physical build: peripherals hang off a shared data bus; two compute hubs (FC + Jetson) drive it; a power bus feeds everything.",
       fs=13, col=PALETTE["muted"])

BUS_Y = 470
# --- power source (top-left) ---
d.box(40,95,180,70,"Smart Battery",["6S–12S Li-ion pack"],"power",fs=12,tfs=13)
d.box(40,185,180,120,"Power Distribution / PMU",
      ["Regulated rails:","  19 V → Jetson","  5 V → FC · GNSS","  12 V → radios · payload"],"power",fs=12)
d.line(130,165,130,185,kind="power",arrow="end")

# --- peripherals (top row) hang DOWN to the bus ---
peri = [
 (260,"nav","Anti-spoof GNSS",["mosaic-X5 / ZED-F9P","+ CRPA antenna"],"UART·RTK / USB·C/N₀"),
 (490,"radio","C2 MANET Radio",["Silvus / Doodle Labs","the monitored link"],"Ethernet (inspected)"),
 (720,"radio","4G/5G + RFD900x",["BVLOS backhaul","backup telemetry"],"USB / UART"),
 (950,"payload","Multispectral Payload",["Altum-PT + DLS 2","NDVI / NDRE"],"GigE + trigger"),
]
for x,dom,title,lines,iface in peri:
    d.box(x,150,215,110,title,lines,dom,fs=12)
    cx=x+107
    d.line(cx,260,cx,BUS_Y,kind="data",arrow="end")
    tw=len(iface)*5.6+8
    d.parts.append(f'<rect x="{cx-tw/2}" y="{(260+BUS_Y)/2-9}" width="{tw}" height="15" rx="3" fill="#fff" opacity="0.95"/>')
    d.text(cx,(260+BUS_Y)/2+2.5,iface,fs=10,col=PALETTE["data"],anchor="middle",bold=True)

# --- the shared data bus ---
d.parts.append(f'<rect x="160" y="{BUS_Y-4}" width="1000" height="8" rx="4" fill="#1f5fbf"/>')
d.text(170,BUS_Y-12,"Onboard data bus  (Ethernet · serial/UART · USB · I²C)",fs=12,bold=True,col="#1f5fbf")

# --- compute hubs + secure element (bottom row) hang UP to the bus ---
d.box(260,560,230,150,"Flight Controller (PX4)",
      ["Pixhawk 6X / Cube Orange+","attitude · GNSS fusion","RTK inject · fail-safe","motor & gimbal PWM"],"flight",fs=12)
d.box(560,560,260,175,"Embedded AI — Jetson Orin NX 16GB",
      ["M4 MambaShield (INT8)","onboard NDVI harvest screen","RobustIDPS.ai detect container","TensorRT · JetPack · MAVSDK"],"onboard",fs=12,tfs=12)
d.box(900,560,215,110,"Secure Element",["Microchip ATECC608","verifies signed models"],"sec",fs=12)
for cx,iface in [(375,"UART · MAVLink"),(690,"Ethernet · USB"),(1007,"I²C")]:
    d.line(cx,560,cx,BUS_Y,kind="data",arrow="both")
    tw=len(iface)*5.6+8
    d.parts.append(f'<rect x="{cx-tw/2}" y="{(560+BUS_Y)/2-9}" width="{tw}" height="15" rx="3" fill="#fff" opacity="0.95"/>')
    d.text(cx,(560+BUS_Y)/2+2.5,iface,fs=10,col=PALETTE["data"],anchor="middle",bold=True)

# --- power lines (left) to the two hubs ---
d.elbow(130,305,260,620,kind="power",label="5 V",midx=175)
d.elbow(130,305,560,650,kind="power",label="19 V",midx=210)
d.text(40,335,"12 V / 5 V rails also feed",fs=10,col=PALETTE["power"])
d.text(40,350,"radios, GNSS & payload",fs=10,col=PALETTE["power"])

# --- RF (wireless) annotations ---
d.text(367,140,"↑ RF to GNSS satellites",fs=10.5,col=PALETTE["rf"],anchor="middle")
d.line(367,150,367,120,kind="rf",arrow="start")
d.text(597,140,"↑ RF to ground C2",fs=10.5,col=PALETTE["rf"],anchor="middle")
d.line(597,150,597,120,kind="rf",arrow="start")

d.legend(900,720,[("power","Power rail"),("data","Wired data bus"),
                  ("rf","RF / wireless link")])
d.text(40,760,"Data-flow note:",fs=12,bold=True)
d.text(40,781,"GNSS C/N₀ + spoof flags and the C2-radio Ethernet stream are fed live into MambaShield (M4) / M6 on the Jetson —",fs=11.5)
d.text(40,799,"the network-layer telemetry the research models classify in real time while the FC keeps flying the RTK survey grid.",fs=11.5)
d.save("diagram1_wiring.svg")

# ==================================================== Diagram 2: 3-tier deploy
d = SVG(1240, 860, "Diagram 2  —  Three-Tier Deployment Architecture")
d.text(30,60,"Where each research model runs, and how models & data move between the UAV, the field gateway, and the cloud.",
       fs=13, col=PALETTE["muted"])
# Tier 1
d.rect(40,100,360,660, fill="#f2fbf6", stroke="#159a6b", rx=14, sw=2)
d.text(60,130,"TIER 1 — ONBOARD (UAV)",fs=15,bold=True,col="#0f7a52")
d.box(70,150,300,120,"Flight Controller (PX4)",["attitude · GNSS fusion","fail-safe · RTK"],"flight")
d.box(70,300,300,150,"Jetson Orin NX 16GB",["M4 MambaShield (INT8)","onboard NDVI harvest flag","lowest-latency detection"],"onboard")
d.box(70,480,300,110,"Sensors / Links",["anti-spoof GNSS","C2 MANET radio","multispectral cam"],"nav")
d.box(70,620,300,90,"Secure Element",["ATECC608 — verifies","signed model updates"],"sec")
# Tier 2
d.rect(440,100,360,660, fill="#eef4fc", stroke="#1f5fbf", rx=14, sw=2)
d.text(460,130,"TIER 2 — EDGE GATEWAY (field)",fs=15,bold=True,col="#1a4fa0")
d.box(470,150,300,180,"Jetson AGX Orin 64GB",
      ["Full RobustIDPS.ai stack","M1 CT/SDE-TGNN","M6 uncertainty calibration","M7 game-theoretic cert."],"edge")
d.box(470,360,300,120,"Agronomy Engine",["Full crop-maturity model","→ per-zone harvest-due map"],"payload")
d.box(470,510,300,110,"Federated Client + RTK",["FL client · RTK base/NTRIP","local dashboard"],"radio")
d.box(470,650,300,80,"HSM (YubiHSM 2)",["signs/verifies PQC updates"],"sec")
# Tier 3
d.rect(840,100,360,660, fill="#eef1f6", stroke="#41506b", rx=14, sw=2)
d.text(860,130,"TIER 3 — CLOUD / REGIONAL",fs=15,bold=True,col="#33415c")
d.box(870,150,300,150,"GPU Server / VM",["FedGTD aggregation","cross-farm model fusion"],"cloud")
d.box(870,330,300,140,"Model Update Service",["PQC-signed (Kyber-1024 /","Dilithium, GOST R 34.10)","versioned rollout"],"cloud")
d.box(870,500,300,130,"Agronomy Dashboards",["harvest-readiness maps","long-term storage & audit"],"cloud")

# flows between tiers
d.line(370,360,470,240,kind="data",label="telemetry graph",arrow="end",lx=420,ly=290)
d.line(470,700,370,665,kind="rf",label="signed update ↓",arrow="end",lx=420,ly=675)
d.line(770,240,870,225,kind="data",label="FL weights ↑",arrow="both",lx=820,ly=210)
d.line(870,400,770,690,kind="rf",label="PQC update ↓",arrow="end",lx=815,ly=560)
d.legend(40, 780, [("data","wired / LAN"),("rf","wireless / WAN")])
d.text(560,800,"Offline-tolerant: Tier 1 flies and detects with no link; Tiers 2–3 sync when connectivity returns.",fs=12,italic=True,col=PALETTE["muted"])
d.save("diagram2_tiers.svg")

# ==================================================== Diagram 3: inference flow
d = SVG(1240, 720, "Diagram 3  —  Dual Inference Pipeline (Harvest + Security)")
d.text(30,60,"One airframe, two co-running workloads: crop harvest-readiness sensing and network-layer robustness.",
       fs=13, col=PALETTE["muted"])
# top lane: harvest
d.text(40,110,"HARVEST-READINESS LANE",fs=13,bold=True,col="#a37a00")
d.box(40,125,200,90,"Multispectral capture",["Altum-PT + DLS 2"],"payload",fs=12)
d.box(280,125,200,90,"Radiometric calib.",["reflectance panel"],"payload",fs=12)
d.box(520,125,210,90,"NDVI / NDRE",["vegetation indices"],"payload",fs=12)
d.box(770,125,210,90,"Maturity model",["per-zone classifier"],"payload",fs=12)
d.box(1015,125,190,90,"Harvest-due map",["due / not / N days"],"field",fs=12)
for x in (240,480,730,980): d.line(x,170,x+40,170,kind="data",arrow="end")
# bottom lane: security
d.text(40,300,"NETWORK-LAYER SECURITY LANE",fs=13,bold=True,col="#1a4fa0")
d.box(40,315,200,100,"Link + GNSS taps",["C2 Ethernet, C/N₀,","spoof flags"],"radio",fs=12)
d.box(280,315,200,100,"M4 MambaShield",["INT8 onboard","real-time detect"],"onboard",fs=12)
d.box(520,315,210,100,"M1 CT/SDE-TGNN",["temporal-graph","anomaly (edge)"],"edge",fs=12)
d.box(770,315,210,100,"M6 + M7",["calibrated trust +","certified policy"],"edge",fs=12)
d.box(1015,315,190,100,"Decision",["continue / re-route /","safe-hold"],"sec",fs=12)
for x in (240,480,730,980): d.line(x,365,x+40,365,kind="data",arrow="end")
# cross link
d.line(340,315,340,215,kind="timing",label="keeps link alive → data keeps flowing",arrow="end",lx=560,ly=270)
# fusion
d.box(440,470,360,120,"Fusion & Uplink (Tier 2/3)",
      ["harvest map + security log → dashboard","FedGTD federated update ↑ · PQC-signed model ↓",
       "field completion vs UAV-EW-Bench-2026"],"cloud",fs=12)
d.line(620,415,620,470,kind="data",arrow="end")
d.line(1110,215,900,470,kind="data",arrow="end",label="",)
d.legend(40, 560, [("data","inference flow"),("timing","cross-lane dependency")])
d.save("diagram3_pipeline.svg")

# ==================================================== Diagram 4: field demo
d = SVG(1240, 780, "Diagram 4  —  Field Test-Run / Demonstration")
d.text(30,60,"Industrial test sortie: survey the field, decide harvest readiness, and validate robustness under measured interference.",
       fs=13,col=PALETTE["muted"])
# sky / field scene
d.rect(40,90,760,470, fill="#f4f9fc", stroke="#c7d2e0", rx=12)
# field zones
zones=[("#bfe3a0","Zone A · ready"),("#e7e08a","Zone B · ~5 days"),("#e8b98a","Zone C · not ready")]
for i,(c,lab) in enumerate(zones):
    x=70+i*240
    d.rect(x,380,220,150, fill=c, stroke="#7a8a5a", rx=8)
    tw=len(lab)*6.6+10
    d.parts.append(f'<rect x="{x+110-tw/2}" y="{398}" width="{tw}" height="17" rx="3" fill="#ffffff" opacity="0.82"/>')
    d.text(x+110,411,lab,fs=12,anchor="middle",bold=True)
d.text(70,370,"Survey grid over the field (RTK-guided)",fs=11.5,italic=True,col=PALETTE["muted"])
# drone
d.rect(360,150,120,54, fill="#eaf7f0", stroke="#159a6b", rx=9, sw=1.8)
d.text(420,183,"UAV",fs=14,anchor="middle",bold=True,col="#0f7a52")
# survey path
d.parts.append('<path d="M 120 340 L 700 340 L 700 300 L 120 300 L 120 260 L 700 260" '
               'fill="none" stroke="#1f5fbf" stroke-width="2" stroke-dasharray="6 5"/>')
d.text(150,250,"NDVI capture sweep",fs=11,col=PALETTE["data"])
# jammer
d.rect(640,430,130,90, fill="#fdeaea", stroke="#c0392b", rx=9, sw=1.8)
d.text(705,455,"EW source",fs=12,anchor="middle",bold=True,col="#c0392b")
d.text(705,474,"(range /",fs=10.5,anchor="middle",col="#c0392b")
d.text(705,489,"Faraday only)",fs=10.5,anchor="middle",col="#c0392b")
d.line(640,455,500,190,kind="rf",label="J/S ↑",arrow="end",lx=545,ly=300)
# ground station
d.rect(830,90,370,300, fill="#eef4fc", stroke="#1f5fbf", rx=12)
d.text(850,120,"GROUND STATION (Tier 2)",fs=14,bold=True,col="#1a4fa0")
d.box(855,135,320,70,"GCS + Edge Gateway",["QGC mission · RobustIDPS.ai live"],"edge",fs=11)
d.box(855,225,150,70,"RTK base",["cm corrections"],"radio",fs=11)
d.box(1025,225,150,70,"Spectrum analyzer",["measures real J/S"],"cloud",fs=11)
d.box(855,315,320,60,"Harvest-due map + security log",[""],"field",fs=11)
d.line(480,175,830,200,kind="rf",label="C2 + telemetry (monitored)",arrow="both",lx=650,ly=175)
# outcomes panel
d.rect(830,410,370,320, fill="#fbfcfe", stroke="#c7d2e0", rx=12)
d.text(850,440,"RUN OUTPUTS",fs=14,bold=True,col="#14213d")
outs=["1  Per-zone harvest-due map (NDVI/NDRE)",
      "2  Security event log (M4/M6/M7 verdicts)",
      "3  Measured completion vs J/S curve",
      "4  Field vs UAV-EW-Bench-2026 comparison",
      "5  Signed audit trail (DO-326A + PQC)"]
for i,o in enumerate(outs):
    d.text(852,470+i*30,o,fs=12)
# procedure strip
d.rect(40,590,760,140, fill="#fbfcfe", stroke="#c7d2e0", rx=12)
d.text(60,616,"SORTIE PROCEDURE",fs=13,bold=True,col="#14213d")
steps=["① Calibrate (panel + RTK + verify signed models)","② Baseline survey → first harvest-due map",
       "③ Characterise RF (spectrum analyzer → J/S)","④ (Authorized) EW stress across J/S sweep",
       "⑤ Export maps + logs; compare to benchmark"]
for i,s in enumerate(steps):
    d.text(60,644+i*17,s,fs=11.5)
d.legend(830, 740-0, [("rf","wireless"),("data","wired")]) if False else None
d.save("diagram4_field.svg")
print("all diagrams done")
