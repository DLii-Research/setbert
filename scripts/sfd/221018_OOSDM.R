#######
options(java.parameters = "-Xmx8g")

library(raster)
library(sp)
library(sf)
library(rcompanion)
library(rgdal)
library(corrplot)
library(MASS)
library(ggplot2)
library(dismo)
library(randomForest)
library(dplyr)
library(ENMeval)

###########
#Data Prep
###########
states<-raster::shapefile("F:/Grisnik/Bias_testing/StatesShp/states_drop.shp")
TN<-states[states$State_Code=="TN",]

dataframe<-read.csv("220312_CMFP.df_ASR.csv")
dataframe<-read.csv("230113_SFDmetadata.formatted_ASR.csv")

spatialdata<-dataframe[,c('gpsn','gpsw', 'state', 'oo_present')]
names(spatialdata)<-c("gpsn", "gpsw",'state', 'oo')


spatialdatatn<-as.data.frame(spatialdata[spatialdata$state=="TN",])

loc<-spatialdatatn[1:2]
summary(loc)
loc<-na.omit(loc)
loc<-loc[c("gpsw","gpsn")]
 
pos<-as.data.frame(spatialdatatn[spatialdatatn$oo==1,])
pos<-pos[1:2]
pos<-na.omit(pos)
pos<-pos[c("gpsw","gpsn")]
names(pos)<-c("lon", "lat")
#write.csv(pos, file="221018_poslocs.csv")


TN<-states[states$State_Code=="TN",]
temp<-pos
temp<-subset(temp, !is.na(lon)&!is.na(lat))
dups=duplicated(temp[,c("lon","lat")])
temp<-temp[!dups,]
sppoints<-temp[,c('lon','lat')] 
sppoints<-SpatialPointsDataFrame(coords=sppoints, data=sppoints, proj4string=CRS("+proj=longlat +datum=WGS84 +ellps+WGS84+towgs84=0,0,0"))

tempcropped<-as.data.frame(sppoints)
min.lon<-floor(min(tempcropped$lon))
max.lon<-ceiling(max(tempcropped$lon))
min.lat<-floor(min(tempcropped$lat))
max.lat<-ceiling(max(tempcropped$lat))
longrid = seq(min.lon,max.lon,0.05)
latgrid = seq(min.lat,max.lat,0.05)
subs = c()
for(j in 1:(length(longrid)-1)){
  for(k in 1:(length(latgrid)-1)){
    gridsq = subset(tempcropped, lat > latgrid[k] & lat < latgrid[k+1] & lon > longrid[j] & lon < longrid[j+1])
    if(dim(gridsq)[1]>0){
      subs = rbind(subs, gridsq[sample(1:dim(gridsq)[1],1 ), ])
    }
  }
}
tempfinal<-subset(subs, select=-c(lon.1,lat.1))

#write.csv(tempfinal, file="230120_spatialfilter.csv")
 
##########
#Landscape Factors
##########
bioclim.data_30_stack<-stack(list.files(path = 'F:/Grisnik/BioClim_data_30/', pattern = '\\.tif$', full.names = T))

states<-raster::shapefile("F:/Grisnik/Bias_testing/StatesShp/states_drop.shp")
TN<-states[states$State_Code=="TN",]

all.equal(wkt(bioclim.data_30_stack),
          wkt(TN))
TN<-spTransform(TN, crs(bioclim.data_30_stack))
all.equal(wkt(bioclim.data_30_stack),
          wkt(TN))

bioclimstack<-crop(bioclim.data_30_stack, TN)
bioclimstack<-mask(bioclimstack, TN)

#for (i in 1:nlayers(bioclimstack)){
#  writeRaster(bioclimstack[[i]], paste0(file= "Envdata/", names(bioclimstack[[i]])), format='GTiff',overwrite=TRUE)
#}

states<-raster::shapefile("F:/Grisnik/Bias_testing/StatesShp/states_drop.shp")
TN<-states[states$State_Code=="TN",]

bioclimstack<-stack(list.files(path = 'Envdata/', pattern = 'bio*', full.names = T))

DEM<-raster("F:/220309_BlackPineSnakes/Data/BPS/DEM/TotalRangeDEM/BPS_DEM.tif")
TN<-spTransform(TN, crs(DEM))

all.equal(wkt(DEM),
          wkt(bioclimstack))

DEMTN<-crop(DEM, TN)
DEMTN<-mask(DEMTN, TN)
crs(DEMTN)<-crs(DEM)
writeRaster(DEMTN, file="Envdata/DEMTN", format='GTiff',overwrite=TRUE)

sand<-raster("F:/220309_BlackPineSnakes/Data/BPS/SSURGO_US/US_sand.tif")

sandTN<-crop(sand, TN)
sandTN<-mask(sandTN, TN)
crs(sandTN)<-crs(sand)

writeRaster(sandTN, file="Envdata/sandTN", format='GTiff',overwrite=TRUE)

clay<-raster("F:/220309_BlackPineSnakes/Data/BPS/SSURGO_US/US_clay.tif")

all.equal(wkt(clay),
          wkt(TN))

clayTN<-crop(clay, TN)
clayTN<-mask(clayTN, TN)
crs(clayTN)<-crs(clay)

writeRaster(clayTN, file="Envdata/clayTN", format='GTiff',overwrite=TRUE)

OM<-raster("F:/220309_BlackPineSnakes/Data/BPS/SSURGO_US/US_OM.tif")

all.equal(wkt(OM),
          wkt(TN))

OMTN<-crop(OM, TN)
OMTN<-mask(OMTN, TN)
crs(OMTN)<-crs(OM)

writeRaster(OMTN, file="Envdata/OMTN", format='GTiff')

Elev<-raster("F:/220309_BlackPineSnakes/Data/BPS/SSURGO_US/US_elev.tif",overwrite=TRUE)

all.equal(wkt(Elev),
          wkt(TN))

ElevTN<-crop(Elev, TN)
ElevTN<-mask(ElevTN, TN)
crs(ElevTN)<-crs(Elev)

writeRaster(ElevTN, file="Envdata/ElevTN", format='GTiff')

hydrogrp<-raster("F:/220309_BlackPineSnakes/Data/BPS/SSURGO_US/US_hydgrp.tif",overwrite=TRUE)

all.equal(wkt(hydrogrp),
          wkt(TN))

hydrogrpTN<-crop(hydrogrp, TN)
hydrogrpTN<-mask(hydrogrpTN, TN)
crs(hydrogrpTN)<-crs(hydrogrp)

writeRaster(hydrogrpTN, file="Envdata/hydrogrpTN", format='GTiff')

soiltax<-raster("F:/220309_BlackPineSnakes/Data/BPS/SSURGO_US/US_taxorder.tif",overwrite=TRUE)

all.equal(wkt(soiltax),
          wkt(TN))

soiltaxTN<-crop(soiltax, TN)
soiltaxTN<-mask(soiltaxTN, TN)
crs(soiltaxTN)<-crs(soiltax)

writeRaster(soiltaxTN, file="Envdata/soiltaxTN", format='GTiff')

drain<-raster("F:/220309_BlackPineSnakes/Data/BPS/SSURGO_US/reclsasdrain1.tif",overwrite=TRUE)

all.equal(wkt(drain),
          wkt(TN))

drainTN<-crop(drain, TN)
drainTN<-mask(drainTN, TN)
crs(drainTN)<-crs(drain)
writeRaster(drainTN, file="Envdata/drainTN", format='GTiff')

floodfreq<-raster("F:/220309_BlackPineSnakes/Data/BPS/SSURGO_US/us_floodfreq.tif",overwrite=TRUE)
TN<-spTransform(TN, crs(floodfreq))

all.equal(wkt(floodfreq),
          wkt(TN))

floodfreqTN<-crop(floodfreq, TN)
floodfreqTN<-mask(floodfreqTN, TN)
crs(floodfreqTN)<-crs(floodfreq)

writeRaster(floodfreqTN, file="Envdata/floodfreqTN", format='GTiff',overwrite=TRUE)

canopycover<-raster("F:/220309_BlackPineSnakes/Data/BPS/Canopycover/cancover.tif")
TN<-spTransform(TN, crs(canopycover))

all.equal(wkt(canopycover),
          wkt(TN))

canopycoverTN<-crop(canopycover, TN)
canopycoverTN<-mask(canopycoverTN, TN)
crs(canopycoverTN)<-crs(canopycover)

writeRaster(canopycoverTN, file="Envdata/canopycoverTN", format='GTiff',overwrite=TRUE)

bioclimstack<-stack(list.files(path = 'Envdata/', pattern = 'bio*', full.names = T))

DEM<-raster("Envdata/DEMTN.tif")
sand<-raster("Envdata/sandTN.tif")
clay<-raster("Envdata/clayTN.tif")
OM<-raster("Envdata/OMTN.tif")
Elev<-raster("Envdata/ElevTN.tif")
hydrogrp<-raster("Envdata/hydrogrpTN.tif")
soiltax<-raster("Envdata/soiltaxTN.tif")
drain<-raster("Envdata/drainTN.tif")
floodfreq<-raster("Envdata/floodfreqTN.tif")
canopycover<-raster("Envdata/canopycoverTN.tif")

compareRaster(DEM,sand, clay, OM, Elev, hydrogrp, soiltax, drain, floodfreq, canopycover,
              extent=FALSE, rowcol=FALSE, crs=TRUE, res=TRUE, orig=FALSE,
              rotation=TRUE, values=FALSE, stopiffalse=TRUE, showwarning=FALSE)      
 
sandprj<-projectRaster(sand,DEM)
clayprj<-projectRaster(clay,DEM)
OMprj<-projectRaster(OM,DEM)
Elevprj<-projectRaster(Elev,DEM)
hydrogrpprj<-projectRaster(hydrogrp,DEM)
soiltaxprj<-projectRaster(soiltax,DEM)
drainprj<-projectRaster(drain,DEM)
floodfreqprj<-projectRaster(floodfreq,DEM)
canopycoverreqprj<-projectRaster(canopycover,DEM)

compareRaster(DEM,sandprj, clayprj, OMprj, Elevprj, hydrogrpprj, soiltaxprj, drainprj, floodfreqprj,
              canopycoverreqprj,
              extent=TRUE, rowcol=TRUE, crs=TRUE, res=TRUE, orig=TRUE,
              rotation=TRUE, values=FALSE, stopiffalse=TRUE, showwarning=FALSE)  

bioclimprj<-projectRaster(bioclimstack,DEM)

compareRaster(DEM,sandprj, clayprj, OMprj, Elevprj, hydrogrpprj, soiltaxprj, drainprj, floodfreqprj,
              canopycoverreqprj,bioclimprj,
              extent=TRUE, rowcol=TRUE, crs=TRUE, res=TRUE, orig=TRUE,
              rotation=TRUE, values=FALSE, stopiffalse=TRUE, showwarning=FALSE)  

Factorstack<-stack(DEM,sandprj, clayprj, OMprj, Elevprj, hydrogrpprj, soiltaxprj, drainprj, floodfreqprj,
                  canopycoverreqprj,bioclimprj)

for (i in 1:nlayers(Factorstack)){
  writeRaster(Factorstack[[i]], paste0(file= "Envdata/Reprj/", names(Factorstack[[i]])), format='GTiff',overwrite=TRUE)
}

envdata<-stack(list.files(path = 'Envdata/Reprj', pattern = '\\.tif$', full.names = T))
states<-raster::shapefile("F:/Grisnik/Bias_testing/StatesShp/states_drop.shp")
TN<-states[states$State_Code=="TN",]
all.equal(wkt(envdata),
          wkt(TN))

TN<-spTransform(TN, crs(envdata))

all.equal(wkt(envdata),
          wkt(TN))

envdata<-stack(list.files(path = 'Envdata/Reprj/', pattern = '\\.tif$', full.names = T))

bioclim<-stack(list.files(path = 'Envdata/Reprj/', pattern = 'bio', full.names = T))
soil<-stack(list.files(path = 'Envdata/Reprj/', pattern = '*TN', full.names = T))

cors_subset_30=cor(values(bioclim),use='complete.obs')
corrplot(cors_subset_30,order = "AOE", addCoef.col = "grey",number.cex=.6) # plot correlations

bioclim_crop=bioclim[[c("bio2_13","bio4_13","bio5_13","bio6_13","bio8_13","bio12_13")]] # keep just reasonably uncorrelated ones
cors_subset_30=cor(values(bioclim_crop),use='complete.obs')
corrplot(cors_subset_30,order = "AOE", addCoef.col = "grey",number.cex=.6) # plot correlations

cors_subset_30=cor(values(soil),use='complete.obs')
corrplot(cors_subset_30,order = "AOE", addCoef.col = "grey",number.cex=.6) # plot correlations

soil_crop=soil[[c("canopycoverTN","clayTN","drainTN","floodfreqTN","hydrogrpTN",
                  "OMTN", "sandTN","soiltaxTN","DEMTN")]] # keep just reasonably uncorrelated ones
cors_subset_30=cor(values(soil_crop),use='complete.obs')
corrplot(cors_subset_30,order = "AOE", addCoef.col = "grey",number.cex=.6) # plot correlations

factors<-stack(bioclim_crop, soil_crop)
cors_subset_30=cor(values(factors),use='complete.obs')
corrplot(cors_subset_30,order = "AOE", addCoef.col = "grey",number.cex=.6) # plot correlations

for (i in 1:nlayers(factors)){
  writeRaster(factors[[i]], paste0(file= "Envdata/Final/", names(factors[[i]])), format='GTiff',overwrite=TRUE)
}

##########
#Bias File
##########
dataframe<-read.csv("220312_CMFP.df_ASR.csv")
spatialdata<-dataframe[,c('gpsn','gpsw', 'state', 'oo')]
DEM<-raster("F:/220309_BlackPineSnakes/Data/BPS/DEM/TotalRangeDEM/BPS_DEM.tif")
DEMcrs<-crs(DEM)
 
spatialdatatn<-as.data.frame(spatialdata[spatialdata$state=="TN",])

loc<-spatialdatatn[1:2]
loc<-rev(loc)
summary(loc)
loc<-na.omit(loc)
names(loc)<-c("lon", "lat")
states<-raster::shapefile("F:/Grisnik/Bias_testing/StatesShp/states_drop.shp")
TN<-states[states$State_Code=="TN",]

locations <- loc
locations<-SpatialPointsDataFrame(coords=locations, data=locations, proj4string=crs(crs(TN)))

plot(TN)
plot(locations, add=T)

envdata<-stack(list.files(path = 'Envdata/Final', pattern = '\\.tif$', full.names = T))

envdata2<-envdata
crs(envdata2)<-crs(DEM)
 
all.equal(wkt(envdata2),
          wkt(DEM))
 
TN<-spTransform(TN, crs(crs(envdata2)))
 
locations <- spTransform(locations,crs(crs(envdata2)))

all.equal(wkt(envdata2),
          wkt(TN),
          wkt(locations))

plot(envdata2$bio12_13)
plot(TN, add=T)
plot(locations, add=T)

climdat30 <- brick(envdata2)
cat(wkt(climdat30))

all.equal(wkt(climdat30),
          wkt(TN),
          wkt(locations))

TN<-spTransform(TN, crs(crs(climdat30)))
locations <- spTransform(locations,crs(crs(climdat30)))

all.equal(wkt(climdat30),
          wkt(TN),
          wkt(locations))

climdat30<-crop(climdat30, TN)
climdat30<-mask(climdat30, TN)

cat(wkt(climdat30))

occur.ras30 <- rasterize(locations, climdat30, 1)
plot(occur.ras30)

occur.states30 <- mask(occur.ras30, TN) %>% crop(TN)
cat(wkt(occur.states30))

plot(occur.states30)

occur.states_DF_30<-as.data.frame(occur.states30, xy=TRUE)
summary(occur.states_DF_30)  
occur.states_DF_30 <- occur.states_DF_30[!is.na(occur.states_DF_30$layer), ]
summary(occur.states_DF_30)

presences30 <- which(values(occur.states30) == 1)
pres.locs30 <- coordinates(occur.states30)[presences30, ]

dens30 <- kde2d(pres.locs30[,1], pres.locs30[,2], n = c(nrow(occur.states30), ncol(occur.states30)))
dens.ras30 <- raster(dens30)
cat(wkt(dens.ras30))
crs(dens.ras30)<-crs(climdat30)

all.equal(wkt(dens.ras30),
          wkt(TN))

cat(wkt(dens.ras30))

plot(dens.ras30)
dens.ras30 <- mask(dens.ras30, TN) %>% crop(TN)
cat(wkt(dens.ras30))

plot(dens.ras30)
plot(TN,add=T)
plot(locations, add=T)

den.ras_DF30<-as.data.frame(dens.ras30, xy=TRUE)
summary(den.ras_DF30)  
den.ras_DF30 <- den.ras_DF30[!is.na(den.ras_DF30$layer), ]
summary(den.ras_DF30)
#writeRaster(dens.ras30, "Bias/BiasFile.tif", overwrite=TRUE)

#############
#Target Background 
#############
bias<-raster("Bias/BiasFile.tif")
states<-raster::shapefile("F:/Grisnik/Bias_testing/StatesShp/states_drop.shp")
TN<-states[states$State_Code=="TN",]

dataframe<-read.csv("230113_SFDmetadata.formatted_ASR.csv")
spatialdata<-dataframe[,c('gpsn','gpsw', 'state', 'oo_present')]
spatialdatatn<-as.data.frame(spatialdata[spatialdata$state=="TN",])

loc<-spatialdatatn[1:2]
loc<-rev(loc)
summary(loc)
loc<-na.omit(loc)
names(loc)<-c("lon", "lat")
n<-nrow(loc) 

locations <- loc
locations<-SpatialPointsDataFrame(coords=locations, data=locations, proj4string=crs(crs(TN)))
plot(TN)
plot(locations, add=T)

envdata<-stack(list.files(path = 'Envdata/Final', pattern = '\\.tif$', full.names = T))

xbiaspoints<-rasterToPoints(bias)
pts <- xbiaspoints[sample(nrow(xbiaspoints),15000, prob = xbiaspoints[,3]),]
ptsdf<-as.data.frame(pts, xy=TRUE)
ptsdf<-subset(ptsdf, select=-c(BiasFile)) 
y <- raster::extract(envdata, ptsdf, sp=TRUE)
ydf<-cbind(y,ptsdf)
removey<-na.omit(ydf)
pts2 <- removey[sample(nrow(removey),10000),]
ptsdf<-subset(pts2, select=c("x","y")) 
#write.csv(ptsdf, file="Bias/221025_MaxentBias.csv", row.names = FALSE)

#####
#RandomForest background
#####
xbiaspoints<-rasterToPoints(bias)
pts <- xbiaspoints[sample(nrow(xbiaspoints),n/0.25, prob = xbiaspoints[,3]),]
ptsdf<-as.data.frame(pts, xy=TRUE)
ptsdf<-subset(ptsdf, select=-c(BiasFile)) 
y <- raster::extract(envdata, ptsdf, sp=TRUE) #r being raster stack, p being points)
ydf<-cbind(y,ptsdf)
removey<-na.omit(ydf)
pts2 <- removey[sample(nrow(removey),n),]
ptsdf<-subset(pts2, select=c("x","y")) 
#write.csv(ptsdf, file="Bias/230120_RFbias.csv", row.names = FALSE)

############################################################################################
#Habitat Model SDM-Maxent
############################################################################################
states<-raster::shapefile("F:/Grisnik/Bias_testing/StatesShp/states_drop.shp")
TN<-states[states$State_Code=="TN",]

sppts<-read.csv("230120_spatialfilter.csv")

locations <- sppts[2:3]
locations<-SpatialPointsDataFrame(coords=locations, data=locations, proj4string=crs(crs(TN)))
plot(TN)
plot(locations, add=T)

envdata<-stack(list.files(path = 'Envdata/Final', pattern = '\\.tif$', full.names = T))
bio12_13<-envdata$bio12_13
bio2_13<-envdata$bio2_13
#bio4_13<-envdata$bio4_13
#bio5_13<-envdata$bio5_13
bio6_13<-envdata$bio6_13
bio8_13<-envdata$bio8_13
#floodfreqTN<-envdata$floodfreqTN
DEMTN<-envdata$DEMTN
canopycoverTN<-envdata$canopycoverTN

envdata<-stack(bio12_13,bio2_13,bio6_13,bio8_13,
               DEMTN,canopycoverTN  )
cors_subset_30=cor(values(envdata),use='complete.obs')
corrplot(cors_subset_30,order = "AOE", addCoef.col = "grey",number.cex=.6) # plot correlations
rm(cors_subset_30)

pts <- spTransform(locations, crs(envdata))
TN <- spTransform(TN, crs(envdata))

plot(TN)
plot(envdata$bio12_13,add=T)
plot(pts,add=T)
#writeOGR(pts, dsn=".", layer="221111_pts", driver="ESRI Shapefile" )

t <- raster::extract(envdata, pts, sp=TRUE) #r being raster stack, p being points)
tdf<-as.data.frame(t) 
tdfNAomit<-na.omit(tdf)
#pts<-tdfNAomit[1:2]
#pts <- spTransform(pts, crs(TN))

biaspts<-read.csv("Bias/221025_MaxentBias.csv")
biaspts<-SpatialPointsDataFrame(coords=biaspts, data=biaspts, proj4string=crs(crs(TN)))
biaspts <- spTransform(biaspts, crs(envdata))

xm<-maxent(x = envdata, p = pts, a=biaspts,
           path="MaxentOutput/230120_Oo_climatemodel",overwrite=TRUE, 
           args=c("replicates=5","replicatetype=crossvalidate","threads=4","responsecurves=TRUE","jackknife=TRUE"))

save(xm, file="MaxentOutput/230120_Oo_climatemodel/outputMaxent.RData")

predict<-dismo::predict(object=xm, 
                        x=envdata, progress="text")

sppts<-as.data.frame(pts)
sppts<-sppts[,3:4]
names(sppts)<-c("lon", "lat")

bg<-as.data.frame(biaspts)
bg<-bg[,3:4]
names(bg)<-c("lon", "lat")

occs.z<-cbind(sppts, raster::extract(envdata,sppts))
bg.z<-cbind(bg, raster::extract(envdata, bg))

e.mx.l<-ENMeval::ENMevaluate(occs=occs.z, bg=bg.z, algorithm='maxnet', parallel=F, numCores=3, 
                                 partitions= 'randomkfold',tune.args=list(fc=c("L","LQ","LQH","H"), rm=c(0.5,1,2,5)))

res<-ENMeval::eval.results(e.mx.l)
#opt.aicc<-res%>% filter(delta.AICc==0)
#dir.create(paste0("ENMeval/", filedb$names[i]))
write.csv(res, file=paste0("ENMeval/results.csv"),row.names = FALSE)
aicmods <- which(e.mx.l@results$AICc == min(na.omit(e.mx.l@results$AICc)))[1] # AIC model
aicmods <- e.mx.l@results[aicmods,]
FC_best <- as.character(aicmods$fc[1])
rm_best <- as.character(aicmods$rm[1])
mod<-paste0("fc.", FC_best,"_rm.",rm_best)

pr <- raster::predict(envdata, e.mx.l@models[[mod]], type = 'cloglog')

writeRaster( pr, file="ENMeval/230419_OOcurrentdist",format='GTiff',overwrite=TRUE)

ocgrp<-e.mx.l@occs.grp
ocpts<-e.mx.l@occs[,1:2] 
tssdf<-cbind(ocgrp, ocpts)
tssbck<-e.mx.l@bg[,1:2]
tssone<-tssdf[ocgrp=="1",][,2:3]
tsstwo<-tssdf[ocgrp=="2",][,2:3]
tssthree<-tssdf[ocgrp=="3",][,2:3]
tssfour<-tssdf[ocgrp=="4",][,2:3]
tssfive<-tssdf[ocgrp=="5",][,2:3]
tssone_extract <- as.vector(raster::extract(pr,  tssone))
tsstwo_extract <- as.vector(raster::extract(pr,  tsstwo))
tssthree_extract <- as.vector(raster::extract(pr,  tssthree))
tssfour_extract <- as.vector(raster::extract(pr,  tssfour))
tssfive_extract <- as.vector(raster::extract(pr,  tssfive))
tssbckgrd_extract <- as.vector(raster::extract(pr,  tssbck))
erf1m <- dismo::evaluate(tssone_extract, tssbckgrd_extract )
erf2m <- dismo::evaluate(tsstwo_extract, tssbckgrd_extract )
erf3m <- dismo::evaluate(tssthree_extract, tssbckgrd_extract )
erf4m <- dismo::evaluate(tssfour_extract, tssbckgrd_extract )
erf5m <- dismo::evaluate(tssfive_extract, tssbckgrd_extract )
save(erf1m, file=paste0("ENMeval/erf1.RData"))
save(erf2m, file=paste0("ENMeval/erf2.RData"))
save(erf3m, file=paste0("ENMeval/erf3.RData"))
save(erf4m, file=paste0("ENMeval/erf4.RData"))
save(erf5m, file=paste0("ENMeval/erf5.RData"))

#load("ENMeval/erf1.RData")
#load("ENMeval/erf2.RData")
#load("ENMeval/erf3.RData") 
#load("ENMeval/erf4.RData") 
#load("ENMeval/erf5.RData") 

mat1m<-erf1m@confusion[which(erf1m@TPR+erf1m@TNR==max(erf1m@TPR+erf1m@TNR)),]
try(mat1m<-mat1m[1,], silent=TRUE)
num1m<-(mat1m[1]*mat1m[4])-(mat1m[3]*mat1m[2])
den1m<-(mat1m[1]+mat1m[3])*(mat1m[2]+mat1m[4])
tss1m<-as.numeric(num1m/den1m)

mat2m<-erf2m@confusion[which(erf2m@TPR+erf2m@TNR==max(erf2m@TPR+erf2m@TNR)),]
try(mat2m<-mat2m[1,], silent=TRUE)
num2m<-(mat2m[1]*mat2m[4])-(mat2m[3]*mat2m[2])
den2m<-(mat2m[1]+mat2m[3])*(mat2m[2]+mat2m[4])
tss2m<-as.numeric(num2m/den2m)

mat3m<-erf3m@confusion[which(erf3m@TPR+erf3m@TNR==max(erf3m@TPR+erf3m@TNR)),]
try(mat3m<-mat3m[1,], silent = TRUE)
num3m<-(mat3m[1]*mat3m[4])-(mat3m[3]*mat3m[2])
den3m<-(mat3m[1]+mat3m[3])*(mat3m[2]+mat3m[4])
tss3m<-as.numeric(num3m/den3m)

mat4m<-erf4m@confusion[which(erf4m@TPR+erf4m@TNR==max(erf4m@TPR+erf4m@TNR)),]
try(mat4m<-mat4m[1,], silent=TRUE)
num4m<-(mat4m[1]*mat4m[4])-(mat4m[3]*mat4m[2])
den4m<-(mat4m[1]+mat4m[3])*(mat4m[2]+mat4m[4])
tss4m<-as.numeric(num4m/den4m)

mat5m<-erf5m@confusion[which(erf5m@TPR+erf5m@TNR==max(erf5m@TPR+erf5m@TNR)),]
try(mat5m<-mat5m[1,], silent=TRUE)
num5m<-(mat5m[1]*mat5m[4])-(mat5m[3]*mat5m[2])
den5m<-(mat5m[1]+mat5m[3])*(mat5m[2]+mat5m[4])
tss5m<-as.numeric(num5m/den5m)
avgtssm<-as.numeric((tss1m+tss2m+tss3m+tss4m+tss5m)/5)
stdev<-c(tss1m , tss2m, tss3m, tss4m, tss5m)
stdv<-sd(stdev)
est.loc <- raster::extract(pr,  sppts)
est.bg <- raster::extract(pr, bg)
ev <- dismo::evaluate(est.loc, est.bg)
thr <- dismo::threshold(ev)
avgthr<-(((thr$spec_sens)+(thr$no_omission)+(thr$prevalence)+(thr$equal_sens_spec)+(thr$sensitivity))/5)
output<-matrix(ncol=2, nrow=7)
output[1,1]<-tss1m
output[2,1]<-tss2m
output[3,1]<-tss3m
output[4,1]<-tss4m
output[5,1]<-tss5m
output[6,1]<-avgtssm
output[7,1]<-stdv
output[1,2]<-erf1m@auc
output[2,2]<-erf2m@auc
output[3,2]<-erf3m@auc
output[4,2]<-erf4m@auc
output[5,2]<-erf5m@auc
output[6,2]<-((erf1m@auc+erf2m@auc+erf3m@auc+erf4m@auc+erf5m@auc)/5)
output[7,2]<-sd(c(erf1m@auc,erf2m@auc,erf3m@auc,erf4m@auc,erf5m@auc))
output2<-matrix(ncol=7, nrow=1)
output2[1,1]<-thr[,1]
output2[1,2]<-thr[,2]
output2[1,3]<-thr[,3]
output2[1,4]<-thr[,4]
output2[1,5]<-thr[,5]
output2[1,6]<-thr[,6]
output2[1,7]<-avgthr
colnames(output)<-c("TSS","AUC")
rownames(output)<-c("1","2","3","4","5", "avg", "st dev")
write.csv(output, file="ENMeval/tss.csv")
colnames(output2)<-c("kappa","spec_sens","no_omission", "prevalence", "equal_sens_spec", "sensitivity", "avg")
write.csv(output2, file="ENMeval/thresholds.csv")
save(e.mx.l, file="ENMeval/outputMaxent.RData")

avgstack<-stack(predict)
avg<-calc(avgstack, fun=mean)
#writeRaster( avg, file="MaxentOutput/230120_currentMaxent",format='GTiff',overwrite=TRUE)

metadata<-read.csv("Maxentoutput/230120_Oo_climatemodel/maxentResults.csv")

F10<-metadata$Fixed.cumulative.value.10.Cloglog.threshold[6]
MTP<-metadata$Minimum.training.presence.Cloglog.threshold[6]
Maxsenspec<-metadata$Maximum.training.sensitivity.plus.specificity.Cloglog.threshold[6]
avgthresh<-(F10+MTP+Maxsenspec)/3

#########
#RandomForest
#########
states<-raster::shapefile("F:/Grisnik/Bias_testing/StatesShp/states_drop.shp")
TN<-states[states$State_Code=="TN",]
DEMTN<-raster("F:/220309_BlackPineSnakes/Data/BPS/DEM/TotalRangeDEM/BPS_DEM.tif")

sppts<-read.csv("230120_spatialfilter.csv")
sppts<-sppts[2:3]
locations <- sppts
locations<-SpatialPointsDataFrame(coords=locations, data=locations, proj4string=crs(crs(TN)))
plot(TN)
plot(locations, add=T)

envdata<-stack(list.files(path = 'Envdata/Final', pattern = '\\.tif$', full.names = T))
bio12_13<-envdata$bio12_13
bio2_13<-envdata$bio2_13
#bio4_13<-envdata$bio4_13
#bio5_13<-envdata$bio5_13
bio6_13<-envdata$bio6_13
bio8_13<-envdata$bio8_13
#floodfreqTN<-envdata$floodfreqTN
DEMTN<-envdata$DEMTN
canopycoverTN<-envdata$canopycoverTN

crs(bio12_13)<-crs(DEMTN)
crs(bio2_13)<-crs(DEMTN)
#crs(bio4_13)<-crs(DEMTN)
#crs(bio5_13)<-crs(DEMTN)
crs(bio6_13)<-crs(DEMTN)
crs(bio8_13)<-crs(DEMTN)
crs(canopycoverTN)<-crs(DEMTN)

envdata<-stack(bio12_13,bio2_13,bio6_13,bio8_13,
               DEMTN,canopycoverTN  )

pts <- spTransform(locations, crs(envdata))
TN <- spTransform(TN, crs(envdata))

plot(TN)
plot(envdata$bio12_13,add=T)
plot(pts,add=T)

t <- raster::extract(envdata, pts, sp=TRUE) #r being raster stack, p being points)
tdf<-as.data.frame(t) 
tdfNAomit<-na.omit(tdf)

biaspts<-read.csv("Bias/230120_RFbias.csv")
biaspts<-SpatialPointsDataFrame(coords=biaspts, data=biaspts, proj4string=crs(crs(TN)))
backg <- spTransform(biaspts, crs(envdata))
plot(backg, add=T)
backg<-as.data.frame(backg, xy=T)
backg<-backg[1:2]
colnames(backg) = c('lon', 'lat')
backg <- backg[sample(nrow(backg),58),]

sppts<-tdfNAomit[9:10]
colnames(sppts)= c('lon','lat')
k<-5
group<-kfold(sppts,k)
group[1:10]
unique(group)
groupb<-kfold(backg,k)
groupb[1:10]
unique(groupb)
pres_train1<-sppts[group !=1, ]
pres_test1<-sppts[group==1,]
backg_train1 <- backg[groupb != 1, ]
backg_test1 <- backg[groupb == 1, ]
testpres1 <- data.frame( raster::extract(envdata, pres_test1) )
testbackg1 <- data.frame( raster::extract(envdata, backg_test1) )
train1 <- rbind(pres_train1, backg_train1)
envtrain1 <- raster::extract(envdata, train1)  
pb_train1 <- c(rep(1, nrow(pres_train1)), rep(0, nrow(backg_train1)))
envtrain1 <- data.frame( cbind(pa=pb_train1, envtrain1))
trf <- tuneRF(envtrain1[, 2:ncol(envtrain1)], envtrain1[, "pa"])
mt <- trf[which.min(trf[,2]), 1]
rf1<-randomForest(envtrain1[, 2:ncol(envtrain1)], envtrain1[, "pa"], ntree=2000,mtry=mt)
Impor1<-rf1[["importance"]]
save(rf1, file=paste0("RandomForestOutput/230420_OoModel/outputRF1.RData"))
pres_train2<-sppts[group !=2, ]
pres_test2<-sppts[group==2,]
backg_train2 <- backg[groupb != 2, ]
backg_test2 <- backg[groupb == 2, ]
testpres2 <- data.frame( raster::extract(envdata, pres_test2) )
testbackg2 <- data.frame( raster::extract(envdata, backg_test2) )
train2 <- rbind(pres_train2, backg_train2)
envtrain2 <- raster::extract(envdata, train2)  
pb_train2 <- c(rep(1, nrow(pres_train2)), rep(0, nrow(backg_train2)))
envtrain2 <- data.frame( cbind(pa=pb_train2, envtrain2) )
trf2 <- tuneRF(envtrain2[, 2:ncol(envtrain2)], envtrain2[, "pa"])
mt2 <- trf2[which.min(trf2[,2]), 1]
rf2<-randomForest(envtrain2[, 2:ncol(envtrain2)], envtrain2[, "pa"], ntree=2000,mtry=mt2)
Impor2<-rf2[["importance"]]
save(rf2, file=paste0("RandomForestOutput/230420_OoModel/outputRF2.RData"))

pres_train3<-sppts[group !=3, ]
pres_test3<-sppts[group==3,]
backg_train3 <- backg[groupb != 3, ]
backg_test3 <- backg[groupb == 3, ]
testpres3 <- data.frame( raster::extract(envdata, pres_test3) )
testbackg3 <- data.frame( raster::extract(envdata, backg_test3) )
train3 <- rbind(pres_train3, backg_train3)
pb_train3 <- c(rep(1, nrow(pres_train3)), rep(0, nrow(backg_train3)))
envtrain3 <- raster::extract(envdata, train3)  
envtrain3 <- data.frame( cbind(pa=pb_train3, envtrain3) )
trf3 <- tuneRF(envtrain3[, 2:ncol(envtrain3)], envtrain3[, "pa"])
mt3 <- trf3[which.min(trf3[,2]), 1]
rf3<-randomForest(envtrain3[, 2:ncol(envtrain3)], envtrain3[, "pa"], ntree=2000,mtry=mt3)
Impor3<-rf3[["importance"]]
save(rf3, file=paste0("RandomForestOutput/230420_OoModel/outputRF3.RData"))

pres_train4<-sppts[group !=4, ]
pres_test4<-sppts[group==4,]
backg_train4 <- backg[groupb != 4, ]
backg_test4 <- backg[groupb == 4, ]
testpres4 <- data.frame( raster::extract(envdata, pres_test4) )
testbackg4 <- data.frame( raster::extract(envdata, backg_test4) )
train4 <- rbind(pres_train4, backg_train4)
envtrain4 <- raster::extract(envdata, train4)  
pb_train4 <- c(rep(1, nrow(pres_train4)), rep(0, nrow(backg_train4)))
envtrain4 <- data.frame( cbind(pa=pb_train4, envtrain4) )
trf4 <- tuneRF(  envtrain4[, 2:ncol(  envtrain4)],   envtrain4[, "pa"])
mt4 <- trf4[which.min(trf4[,2]), 1]
rf4<-randomForest(  envtrain4[, 2:ncol(  envtrain4)],   envtrain4[, "pa"], ntree=2000,mtry=mt4)
Impor4<-rf4[["importance"]]
save(rf4, file=paste0("RandomForestOutput/230420_OoModel/outputRF4.RData"))

pres_train5<-sppts[group !=5, ]
pres_test5<-sppts[group==5,]
backg_train5 <- backg[groupb != 5, ]
backg_test5 <- backg[groupb == 5, ]
testpres5 <- data.frame( raster::extract(envdata, pres_test5) )
testbackg5 <- data.frame( raster::extract(envdata, backg_test5) )
train5 <- rbind(pres_train5, backg_train5)
envtrain5 <- raster::extract(envdata, train5)  
pb_train5 <- c(rep(1, nrow(pres_train5)), rep(0, nrow(backg_train5)))
envtrain5 <- data.frame( cbind(pa=pb_train5, envtrain5) )
trf5 <- tuneRF(  envtrain5[, 2:ncol(  envtrain5)],   envtrain5[, "pa"])
mt5 <- trf5[which.min(trf5[,2]), 1]
rf5<-randomForest(  envtrain5[, 2:ncol(  envtrain5)],   envtrain5[, "pa"], ntree=2000,mtry=mt5)
Impor5<-rf5[["importance"]]
save(rf5, file=paste0("RandomForestOutput/230420_OoModel/outputRF5.RData"))

pr1 <- predict(envdata, rf1,na.rm=TRUE, progress="text")
pr2 <- predict(envdata, rf2,na.rm=TRUE, progress="text")
pr3 <- predict(envdata, rf3,na.rm=TRUE, progress="text")
pr4 <- predict(envdata, rf4,na.rm=TRUE, progress="text")
pr5 <- predict(envdata, rf5,na.rm=TRUE, progress="text")

erf1 <- evaluate(testpres1, testbackg1, rf1)
erf2 <- evaluate(testpres2, testbackg2, rf2)
erf3 <- evaluate(testpres3, testbackg3, rf3)
erf4 <- evaluate(testpres4, testbackg4, rf4)
erf5 <- evaluate(testpres5, testbackg5, rf5)
save(erf1, file="RandomForestOutput/230420_OoModel/erf1.RData")
save(erf2, file="RandomForestOutput/230420_OoModel/erf2.RData")
save(erf3, file="RandomForestOutput/230420_OoModel/erf3.RData")
save(erf4, file="RandomForestOutput/230420_OoModel/erf4.RData")
save(erf5, file="RandomForestOutput/230420_OoModel/erf5.RData")

#load("RandomForestOutput/230420_OoModel/erf1.RData")
#load("RandomForestOutput/230420_OoModel/erf2.RData")
#load("RandomForestOutput/230420_OoModel/erf3.RData")
#load("RandomForestOutput/230420_OoModel/erf4.RData")
#load("RandomForestOutput/230420_OoModel/erf5.RData")

mat1m<-erf1@confusion[which(erf1@TPR+erf1@TNR==max(erf1@TPR+erf1@TNR)),]
try(mat1m<-mat1m[1,], silent=TRUE)
num1m<-(mat1m[1]*mat1m[4])-(mat1m[3]*mat1m[2])
den1m<-(mat1m[1]+mat1m[3])*(mat1m[2]+mat1m[4])
tss1m<-as.numeric(num1m/den1m)
mat2m<-erf2@confusion[which(erf2@TPR+erf2@TNR==max(erf2@TPR+erf2@TNR)),]
try(mat2m<-mat2m[1,], silent=TRUE)
num2m<-(mat2m[1]*mat2m[4])-(mat2m[3]*mat2m[2])
den2m<-(mat2m[1]+mat2m[3])*(mat2m[2]+mat2m[4])
tss2m<-as.numeric(num2m/den2m)
mat3m<-erf3@confusion[which(erf3@TPR+erf3@TNR==max(erf3@TPR+erf3@TNR)),]
try(mat3m<-mat3m[1,], silent = TRUE)
num3m<-(mat3m[1]*mat3m[4])-(mat3m[3]*mat3m[2])
den3m<-(mat3m[1]+mat3m[3])*(mat3m[2]+mat3m[4])
tss3m<-as.numeric(num3m/den3m)
mat4m<-erf4@confusion[which(erf4@TPR+erf4@TNR==max(erf4@TPR+erf4@TNR)),]
try(mat4m<-mat4m[1,], silent=TRUE)
num4m<-(mat4m[1]*mat4m[4])-(mat4m[3]*mat4m[2])
den4m<-(mat4m[1]+mat4m[3])*(mat4m[2]+mat4m[4])
tss4m<-as.numeric(num4m/den4m)
mat5m<-erf5@confusion[which(erf5@TPR+erf5@TNR==max(erf5@TPR+erf5@TNR)),]
try(mat5m<-mat5m[1,], silent=TRUE)
num5m<-(mat5m[1]*mat5m[4])-(mat5m[3]*mat5m[2])
den5m<-(mat5m[1]+mat5m[3])*(mat5m[2]+mat5m[4])
tss5m<-as.numeric(num5m/den5m)
avgtssm<-as.numeric((tss1m+tss2m+tss3m+tss4m+tss5m)/5)
stdev<-c(tss1m , tss2m, tss3m, tss4m, tss5m)
stdv<-sd(stdev)

erf1AUC<-erf1@auc
erf2AUC<-erf2@auc
erf3AUC<-erf3@auc
erf4AUC<-erf4@auc
erf5AUC<-erf5@auc
avgAUC<-(erf1AUC+erf2AUC+erf3AUC+erf4AUC+erf5AUC)/5
x<-matrix(ncol=5, nrow=7)
x[1,1]<-1
x[2,1]<-2
x[3,1]<-3
x[4,1]<-4
x[5,1]<-5
x[6,1]<-"avg"
x[7,1]<-"Std Dev"
x[1,2]<-erf1AUC
x[2,2]<-erf2AUC
x[3,2]<-erf3AUC 
x[4,2]<-erf4AUC 
x[5,2]<-erf5AUC 
sd<-sd(x[,2],na.rm=T)
x[7,2]<-sd
x[6,2]<-avgAUC 
trss1 <- threshold(erf1, 'spec_sens')
trss2 <- threshold(erf2, 'spec_sens')
trss3 <- threshold(erf3, 'spec_sens')
trss4 <- threshold(erf4, 'spec_sens')
trss5 <- threshold(erf5, 'spec_sens')
avgtrCss<-(trss1+trss2+trss3+trss4+trss5)/5
x[1,3]<-trss1
x[2,3]<-trss2
x[3,3]<-trss3 
x[4,3]<-trss4 
x[5,3]<-trss5 
x[6,3]<-avgtrCss 
tress1 <- threshold(erf1, 'equal_sens_spec')
tress2 <- threshold(erf2, 'equal_sens_spec')
tress3 <- threshold(erf3, 'equal_sens_spec')
tress4 <- threshold(erf4, 'equal_sens_spec')
tress5 <- threshold(erf5, 'equal_sens_spec')
avgtrCess<-(tress1+tress2+tress3+tress4+tress5)/5
x[1,4]<-tress1
x[2,4]<-tress2
x[3,4]<-tress3 
x[4,4]<-tress4 
x[5,4]<-tress5 
x[6,4]<-avgtrCess
x[1,5]<-tss1
x[2,5]<-tss2
x[3,5]<-tss3 
x[4,5]<-tss4 
x[5,5]<-tss5 
sdtss<-sd(x[1:5,5],na.rm=T)
x[7,5]<-sdtss 
x[6,5]<-avgtss 
x<-data.frame(x)
colnames(x)<-c("Iteration", "AUC", "Maximum sensitivity plus specificity","equal sensitivity and specificity","TSS")

#write.csv(x, file=paste0("RandomForestOutput/230420_OoModel/RFoutput.csv"))
avgstackcurr<-stack(pr1,pr2,pr3,pr4,pr5)
avgcurr<-calc(avgstackcurr, fun=mean)
#writeRaster( avgcurr, file="RandomForestOutput/230420_OoModel/230420_OoModel_RF",format='GTiff',overwrite=TRUE)

#######
#
######
avgstackcurr<-stack(pr1,pr2,pr3,pr4,pr5)
avgcurr<-calc(avgstackcurr, fun=mean)
#writeRaster( avgcurr, file=paste0("RandomForestOutput/OoModel/OoModel_RF"),format='GTiff',overwrite=TRUE)

maxent<-raster("ENMeval/230419_OOcurrentdist.tif")
RF<-raster("RandomForestOutput/OoModel/OoModel_RF.tif")

stack<-stack(maxent, RF)
avgcurr<-calc(stack, fun=mean)
#writeRaster( avgcurr, file="230419_OoModel_avg",format='GTiff',overwrite=TRUE)
DEMTN<-raster("F:/220309_BlackPineSnakes/Data/BPS/DEM/TotalRangeDEM/BPS_DEM.tif")
avg<-raster("230419_OoModel_avg.tif")
all.equal(wkt(DEMTN),
          wkt(avg))

crs(avg)<-crs(DEMTN)
bioclim.data_30_stack<-stack(list.files(path = 'F:/Grisnik/BioClim_data_30/', pattern = '\\.tif$', full.names = T))
resavg<-res(avg)
test<-projectRaster(avg, crs=crs(bioclim.data_30_stack), res=resavg)
#writeRaster(test, file="230419_OoModel_avg_reprj", format='GTiff', overwrite=T)

dataframe<-read.csv("220312_CMFP.df_ASR.csv")
spatialdata<-dataframe[,c('gpsn','gpsw', 'state', 'oo','swab_label')]
names(spatialdata)<-c("lat", "lon",  'state', 'oo','swab_label')
spatialdatatn<-as.data.frame(spatialdata[spatialdata$state=="TN",])
states<-raster::shapefile("F:/Grisnik/Bias_testing/StatesShp/states_drop.shp")
TN<-states[states$State_Code=="TN",]
spatialdatatn<-na.omit(spatialdatatn)

sppts<-spatialdatatn[1:2] 
sppts<-rev(sppts)
sppts<-na.omit(sppts)
locations <- sppts

locations<-SpatialPointsDataFrame(coords=locations, data=locations, proj4string=crs(crs(TN)))
plot(TN)
plot(locations, add=T)

envdata<-stack(list.files(path = 'Envdata/Final', pattern = '\\.tif$', full.names = T))
bio12_13<-envdata$bio12_13
bio2_13<-envdata$bio2_13
#bio4_13<-envdata$bio4_13
#bio5_13<-envdata$bio5_13
bio6_13<-envdata$bio6_13
bio8_13<-envdata$bio8_13
#floodfreqTN<-envdata$floodfreqTN
DEMTN<-envdata$DEMTN
canopycoverTN<-envdata$canopycoverTN

envdata<-stack(bio12_13,bio2_13,bio6_13,bio8_13,
               DEMTN,canopycoverTN  )

pts <- spTransform(locations, crs(envdata))
TN <- spTransform(TN, crs(envdata))
avg<-raster("230419_OoModel_avg.tif")

plot(TN)
plot(envdata$bio12_13,add=T)
plot(pts,add=T)

all.equal(wkt(envdata$bio12_13),
          wkt(avg))
plot(TN)
plot(avg,add=T)
plot(pts,add=T)

y <- raster::extract(avg, pts, sp=TRUE) #r being raster stack, p being points)
ydf<-as.data.frame(y)

tdf<-merge(ydf,spatialdatatn, by="lon")
#write.csv(tdf, "230419_suitabiltyscores.csv")

scores<-read.csv("230419_suitabiltyscores.csv")
sites<-dataframe[,c('site','swab_label')]
df<-merge(scores, sites, by="swab_label")
            
sppts<-read.csv("230120_spatialfilter.csv")
sppts<-sppts[2:3]
states<-raster::shapefile("F:/Grisnik/Bias_testing/StatesShp/states_drop.shp")
TN<-states[states$State_Code=="TN",]

locations<-SpatialPointsDataFrame(coords=sppts, data=sppts, proj4string=crs(crs(TN)))
plot(TN)
plot(locations, add=T)

envdata<-stack(list.files(path = 'Envdata/Final', pattern = '\\.tif$', full.names = T))
bio12_13<-envdata$bio12_13
bio2_13<-envdata$bio2_13
#bio4_13<-envdata$bio4_13
#bio5_13<-envdata$bio5_13
bio6_13<-envdata$bio6_13
bio8_13<-envdata$bio8_13
#floodfreqTN<-envdata$floodfreqTN
DEMTN<-envdata$DEMTN
canopycoverTN<-envdata$canopycoverTN

envdata<-stack(bio12_13,bio2_13,bio6_13,bio8_13,
               DEMTN,canopycoverTN  )

pts <- spTransform(locations, crs(envdata))
TN <- spTransform(TN, crs(envdata))
avg<-raster("230419_OoModel_avg.tif")

plot(TN)
plot(envdata$bio12_13,add=T)
plot(pts,add=T)

all.equal(wkt(envdata$bio12_13),
          wkt(avg))
plot(TN)
plot(avg,add=T)
plot(pts,add=T)

y <- raster::extract(avg, pts, sp=TRUE) #r being raster stack, p being points)
ydf<-as.data.frame(y)

tdf<-merge(ydf,spatialdatatn, by="lon")




