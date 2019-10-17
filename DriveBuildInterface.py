from pathlib import Path

from drivebuildclient.AIExchangeService import AIExchangeService
from drivebuildclient.aiExchangeMessages_pb2 import SimulationID, VehicleID, SimStateResponse, DataRequest, Control

import model


from threading import Thread
from typing import List


class MyAI:

    def __init__(self) -> None:
        pass

    def start(self, sid: SimulationID, vid: VehicleID) -> None:

        service = AIExchangeService("defender.fim.uni-passau.de", 8383)

        while True:
            print(sid.sid + ": Test status: " + service.get_status(sid))
            # Wait for the simulation to request this AI
            sim_state = service.wait_for_simulator_request(sid, vid)
            if sim_state is SimStateResponse.SimState.RUNNING:  # Check whether simulation is still running
                # Request data this AI needs
                request = DataRequest()
                request.request_ids.extend(["egoSpeed", "egoFrontCamera"])  # Add all IDs of data to be requested
                data = service.request_data(sid, vid, request)  # Request the actual data

                # Calculate commands controlling the car
                control = Control()
                speed = data['egoSpeed']
                throttle = 0.03
                if speed < (6 / 3.6):
                    throttle = 0.06
                control.avCommand.accelerate = throttle
                control.avCommand.steer = model.get_command(data['egoFrontCamera'])

                service.control(sid, vid, control)
            else:
                print(sid.sid + ": The simulation is not running anymore (Final state: "
                      + SimStateResponse.SimState.Name(sim_state) + ").")
                print(sid.sid + ": Final test result: " + service.get_result(sid))
                # Clean up everything you have to
                break



if __name__ == "__main__":

    service = AIExchangeService("defender.fim.uni-passau.de", 8383)

    # Send tests
    d1 = Path("/home/heintimo/PycharmProjects/StefanEx/exCriteria.dbc.xml")
    d2 = Path("/home/heintimo/PycharmProjects/StefanEx/exEnvironment.dbe.xml")
    sids = service.run_tests("TimoHeinrich", "rXMjvb7YK", d1, d2)

