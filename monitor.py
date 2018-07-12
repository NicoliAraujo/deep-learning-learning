from robobrowser import RoboBrowser

import time
import smtplib
import re
import datetime

login_url = "https://centraldesistemas.sbc.org.br/ecos/bracis2018"
url_scrape = "https://centraldesistemas.sbc.org.br/ecos/bracis2018/edit"

payload = {
    "data[User][username]": "nicoliparaujo@outlook.com",
    "data[User][password]": "b1tchpl3ase",
}
pwd1 = 'b1tchpl3ase'
while True:
    browser=RoboBrowser()
    browser.open(login_url, verify=False)
    
    form=browser.get_form(action='/login')
    form['data[User][username]'] = payload['data[User][username]']
    form['data[User][password]'] = payload['data[User][password]']
    
    browser.submit_form(form)

    browser.open(url_scrape)
    
    
    qtd_undergraduate = len([m.start() for m in re.finditer('Undergraduate', str(browser.parsed()))])
    
    if qtd_undergraduate<=0:
        time.sleep(300)
        print('ultima checagem: ',datetime.datetime.now())
        continue

    else:
        msg = 'Subject: Inscrições do BRACIS abertas para alunos de graduação! \n\n Atenção, o script que monitora o formulário de inscrição no BRACIS detectou a palavra "Undergraduate" na página!\n Inscreva-se imediatamente em https://centraldesistemas.sbc.org.br/ecos/bracis2018'
        fromaddr = "nicoli.pinheiro@outlook.com.br"
        toaddrs = ['npda.eng@uea.edu.br', 'gol.eng@uea.edu.br', 'lgce.eng@uea.edu.br', 'rass.eng@uea.edu.br',                  'eos.eng@uea.edu.br', 'jnl.eng@uea.edu.br', 'lefb.eng@uea.edu.br', 'mwvda.eng@uea.edu.br', 'julio.costa@ccc.ufcg.edu.br', 'rgs.snf17@uea.edu.br']

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
