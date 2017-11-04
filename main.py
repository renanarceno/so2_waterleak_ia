import tornado.ioloop
import tornado.web
import hidro_neural as neural
import csv
import time


def get_pessoas_default():
    original_filename = "hidro2.csv"
    with open(original_filename, "r") as file:
        reader = csv.DictReader(file)
        now = int(time.time())
        now = now % (24 * 60 * 60)
        for row in reader:
            row_seg = int(row['t_seg'])
            if (now - 20) <= row_seg <= (now + 20):
                return int(row['pessoas'])


class MainHandler(tornado.web.RequestHandler):
    def data_received(self, chunk):
        pass

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
        maquinas = int(self.get_query_argument("maquinas", default=0))
        vazao_total = float(self.get_query_argument("vazao_total", default=0))
        vazao_1 = float(self.get_query_argument("vazao_1", default=0))
        sensor_p1 = int(self.get_query_argument("sensor_p1", default=0))
        vazao_2 = float(self.get_query_argument("vazao_2", default=0))
        sensor_p2 = int(self.get_query_argument("sensor_p2", default=0))
        vazao_3 = float(self.get_query_argument("vazao_3", default=0))
        sensor_p3 = int(self.get_query_argument("sensor_p3", default=0))
        vazamento = int(self.get_query_argument("vazamento", default=0))

        if pessoas == -1:
            pessoas = get_pessoas_default()

        print(pessoas)

        loss, prediction = neural.predict_value(pessoas, maquinas, vazao_total, vazao_1, sensor_p1, vazao_2, sensor_p2, vazao_3, sensor_p3, vazamento)
        vazando = abs(round(prediction))
        if vazando == 0:
            self.write("<font color=\"blue\">NÃO</font> está com vazamento")
        else:
            self.write("<font color=\"red\">ESTÁ</font> com vazamento")
        self.write("<br>")
        self.write("Entropia rede neural: " + str(loss))


def make_app():
    return tornado.web.Application([
        (r"/prediction", MainHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8080)
    print("Starting server")
    tornado.ioloop.IOLoop.current().start()
