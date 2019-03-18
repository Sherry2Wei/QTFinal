
require(data.table)
require(RPostgres)

wrds=dbConnect(Postgres(), 
          host='wrds-pgdata.wharton.upenn.edu',
          port=9737,
          user='',
          password='',
          sslmode='require',
          dbname='wrds')

q='select * from crsp.msedist' # this is a SQL statement
res=dbGetQuery(wrds,q) # run the query
setDT(res) ->res
dbDisconnect(wrds)

require(feather)

write_feather(res,'C:/Dropbox/teaching2018_2019/data_for_quant_trading/msedist.feather')
