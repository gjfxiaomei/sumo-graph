if __name__ == "__main__":
    with open("demand.rou.xml","w") as demand:
        print("""<routes>
            <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" maxSpeed="70" guiShape="passenger"/>
            <route id="up_route" edges="S2TL TL2N"/>
            <route id="left_route" edges="E2TL TL2W"/>""",file=demand)
        for i in range(3600):
            if i % 30 == 0:
                print(' <flow id= "up_flow_%i" type="car" begin="%i" end="%i" number="20" route="up_route"  departLane="best" departSpeed="max"/>'%( i, i, i+10),file=demand)
            if i % 100 == 0:
                print(' <flow id= "left_flow1_%i" type="car" begin="%i" end="%i" number="10" route="left_route"  departLane="best" departSpeed="max"/>'%(i,i,i+10),file=demand)
            if i % 100 == 60:
                print(' <flow id= "left_flow2_%i" type="car" begin="%i" end="%i" number="30" route="left_route"  departLane="best" departSpeed="max"/>'%(i, i, i+10),file=demand)
        print("</routes>",file=demand)