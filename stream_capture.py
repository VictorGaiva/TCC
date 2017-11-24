import requests
import datetime
from time import sleep
from threading import Thread

def CapStream(stream_url, radioName):
    print("Iniciando captura de ", radioName)
    start = datetime.datetime.now()    
    r = requests.get(stream_url, stream=True)
    fileName = '{0}-{1}-{2} {3}-{4}-{5} - '.format(start.year, start.month, start.day, start.hour, start.minute, start.second)
    output_file = open('./audio/radio_cap/'+fileName+radioName+'.wav', 'wb')
    try:
        i = 0
        for block in r.iter_content(1024):
            #a cada 128 blocos escritos
            if(i == 128):
                i = 0
                #se 1 hora tiver se passado
                diff = datetime.datetime.now() - start
                if(diff.seconds >= 3600):
                    #fecha arquivo atual e abre novo
                    start = datetime.datetime.now()
                    output_file.close()
                    fileName = '{0}-{1}-{2} {3}-{4}-{5} - '.format(start.year, start.month, start.day, start.hour, start.minute, start.second)
                    output_file = open('./audio/radio_cap/'+fileName+radioName+'.wav', 'wb')
            else:
                i = i + 1
            output_file.write(block)
    except KeyboardInterrupt:
        pass
    print(radioName, " error")
jobs = []
jobs.append(Thread(target=CapStream, args=("http://radiostream.ufms.br:8000/radioufms", "radioufms")))
#jobs.append(Thread(target=CapStream, args=("http://184.154.195.202:9050", "uniderpfm")))
#jobs.append(Thread(target=CapStream, args=("http://184.154.195.202:9072", "capitalfm")))
#jobs.append(Thread(target=CapStream, args=("http://ice.paineldj3.com.br:8006/live", "metropolitana")))
#jobs.append(Thread(target=CapStream, args=("http://transamerica.crossradio.com.br:9100/live.aac", "transamerica")))
#jobs.append(Thread(target=CapStream, args=("http://17483.live.streamtheworld.com:3690/JP_SP_FM_SC", "jovempan")))
jobs.append(Thread(target=CapStream, args=("https://listen.shoutcast.com/89fmaradiorock", "radiorocksp")))
jobs.append(Thread(target=CapStream, args=("http://174-36-1-94.webnow.net.br/dumont.aac", "radiodumond")))

for t in jobs:
    t.daemon = True
    t.start()

#realize a captura do stream durante 24*7 horas
sleep(60*60*24*7)
#sleep(60)
