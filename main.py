import tornado.ioloop
import tornado.web
import hidro_neural as neural
import csv
import time
import json
import sqlite3 as db


def get_int_default_from_file():
    original_filename = "hidro2.csv"
    with open(original_filename, "r") as file:
        reader = csv.DictReader(file)
        now = int(time.time())
        now = now % (24 * 60 * 60)
        for row in reader:
            row_seg = int(row['t_seg'])
            if (now - 20) <= row_seg <= (now + 20):
                return int(row['pessoas']), int(row['maquinas'])


def check_vazamento(maquinas, pessoas, sensor_p1, sensor_p2, sensor_p3, vazamento, vazao_1, vazao_2, vazao_3, vazao_total):
    loss, prediction = neural.predict_value(pessoas, maquinas, vazao_total, vazao_1, sensor_p1, vazao_2, sensor_p2, vazao_3, sensor_p3, vazamento)
    vazando = abs(round(prediction))
    return loss, vazando


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        """
            http://localhost:8080/
                prediction?
                    pessoas=53
                    &maquinas=2
                    &vazao_total=2.27
                    &vazao_1=0.78
                    &sensor_p1=0
                    &vazao_2=0.78
                    &sensor_p2=0
                    &vazao_3=0.78
                    &sensor_p3=0
                    &vazamento=0
        """
        pessoas = int(self.get_query_argument("pessoas", default=-1))
        maquinas = int(self.get_query_argument("maquinas", default=-1))
        vazao_total = float(self.get_query_argument("vazao_total", default=0))
        vazao_1 = float(self.get_query_argument("vazao_1", default=0))
        sensor_p1 = int(self.get_query_argument("sensor_p1", default=0))
        vazao_2 = float(self.get_query_argument("vazao_2", default=0))
        sensor_p2 = int(self.get_query_argument("sensor_p2", default=0))
        vazao_3 = float(self.get_query_argument("vazao_3", default=0))
        sensor_p3 = int(self.get_query_argument("sensor_p3", default=0))
        vazamento = int(self.get_query_argument("vazamento", default=0))

        if pessoas == -1 or maquinas == -1:
            pessoas_aux, maquinas_aux = get_int_default_from_file()
            if pessoas == -1:
                pessoas = pessoas_aux
            if maquinas == -1:
                maquinas = maquinas_aux

        loss, vazando = check_vazamento(maquinas, pessoas, sensor_p1, sensor_p2, sensor_p3, vazamento, vazao_1, vazao_2, vazao_3, vazao_total)
        if vazando == 0:
            self.write("<font color=\"blue\">NÃO</font> está com vazamento")
        else:
            self.write("<font color=\"red\">ESTÁ</font> com vazamento")
        self.write("<br>")
        self.write("Entropia rede neural: " + str(loss))


class PostHandler(tornado.web.RequestHandler):
    def post(self):
        data = json.loads(self.request.body.decode('utf-8'))
        con = db.connect('water_flow')
        con.executemany("INSERT INTO sensor_data(data, x, y, z, valor, unidade) VALUES (?,?,?,?,?,?)", (data['data'], data['x'], data['y'], data['z'], data['valor'], data['unidade']))
        con.commit()
        con.close()


class CheckHandler(tornado.web.RequestHandler):
    def get(self):
        con = db.connect('water_flow.db.db')
        cursor = con.cursor()
        cursor.execute("SELECT * FROM sensor_data WHERE checado = 0 ORDER BY _id DESC LIMIT 7")  # in our example, theres 7 sensors
        sensor_values = []
        for entry in cursor.fetchall():
            sensor_values.append(entry[5])  # 5 = valor
        pessoas_aux, maquinas_aux = get_int_default_from_file()
        loss, vazando = check_vazamento(maquinas_aux, pessoas_aux, sensor_values[0], sensor_values[1], 0, sensor_values[2], sensor_values[3], sensor_values[4], sensor_values[5], sensor_values[6])
        con.close()
        if vazando == 0:
            self.write("<font color=\"blue\">NÃO</font> está com vazamento")
        else:
            self.write("<font color=\"red\">ESTÁ</font> com vazamento")
        self.write("<br>")
        self.write("Entropia rede neural: " + str(loss))


def make_app():
    return tornado.web.Application([
        (r"/prediction", MainHandler),
        (r"/put", PostHandler),
        (r"/check", CheckHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8080)
    print("Server started")
    tornado.ioloop.IOLoop.current().start()
