from robobrowser import RoboBrowser

import time
import smtplib
import re
import datetime
import difflib

url = "https://bracis2018.mybluemix.net/registration.html"

browser=RoboBrowser()
browser.open(url, verify=False)
pag_ant = str(browser.parsed())

pwd1 = 'b1tchpl3ase'
while True:
    browser=RoboBrowser()
    browser.open(url, verify=False)

    pag_atual = str(browser.parsed())
    
    if pag_atual == pag_ant:
        time.sleep(300)
        print('ultima checagem: ',datetime.datetime.now())
        continue

    else:
        difnovo = ''
        s = difflib.SequenceMatcher(None, pag_ant, pag_atual)
        for block in s.get_matching_blocks():
            difnovo += pag_atual.replace(pag_atual[block.b:block.b+block.size], "")
        msg = 'Subject: Alterações na Página "Registration" do site do BRACIS! \n\n Atenção, o script que monitora a página de informações sobre inscrições no site do BRACIS detectou mudanças:\n\n\n'+difflib+'n\n Acesse agora mesmo para visualizar as alterações!! \n https://bracis2018.mybluemix.net/registration.html'
        fromaddr = "nicoli.pinheiro@outlook.com.br"
        #toaddrs = ['npda.eng@uea.edu.br', 'gol.eng@uea.edu.br', 'lgce.eng@uea.edu.br', 'rass.eng@uea.edu.br', 'eos.eng@uea.edu.br', 'jnl.eng@uea.edu.br', 'lefb.eng@uea.edu.br', 'mwvda.eng@uea.edu.br', 'julio.costa@ccc.ufcg.edu.br', 'rgs.snf17@uea.edu.br']
        toaddrs = ['npda.eng@uea.edu.br']
        server = smtplib.SMTP('smtp-mail.outlook.com', 587)
        server.ehlo()
        server.starttls()
        server.login(fromaddr,pwd1)

        print('From: ', fromaddr)
        print('To: ', toaddrs)
        print('Message: ', msg)

        server.sendmail(fromaddr, toaddrs, msg.encode("utf8"))
        server.quit()
        break
