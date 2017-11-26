#!/usr/bin/env python3
import os
import sys
import time
import serial
import ast
import requests

URL = "http://localhost:8080/put"
DEV = '/dev/ttyACM0'
TIMEOUT = 1000


def init_mote():
    ok = False
    mote_l = None
    while not ok:
        try:
            print("Waiting for " + DEV + " to appear")
            while not os.path.exists(DEV) or not os.access(DEV, os.W_OK):
                pass
            mote_l = serial.Serial(DEV, 115200, timeout=TIMEOUT, write_timeout=10)
            ok = True
        except KeyboardInterrupt:
            raise
        except Exception as exp:
            print("Exception caught: ", exp, file=sys.stderr)
            time.sleep(3)

    print("Mote opened trying to write epoch ", file=sys.stderr)
    ts = bytes(str(int(time.time() * 1000000)), 'ascii')
    try:
        mote_l.write(ts + b'X')
        print("Epoch written ", file=sys.stderr)
    except KeyboardInterrupt:
        raise
    except serial.serialutil.SerialTimeoutException:
        pass

    print("init_mote() done ", file=sys.stderr)
    return mote_l


if __name__ == "__main__":
    mote = init_mote()

    while True:
        data = ''
        try:
            data = mote.read_all()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("Exception caught:", e, file=sys.stderr)
        if not len(data):
            mote.close()
            mote = init_mote()
        else:
            # SEND TO SERVER
            decodedData = data.decode()
            for line in decodedData.splitlines():
                try:
                    dic = ast.literal_eval(line)
                    if dic is dict:
                        response = requests.post(URL, data=dic)
                        print(response.status_code, response.reason)
                        pass
                except Exception as ignore:
                    pass
