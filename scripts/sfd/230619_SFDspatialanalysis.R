# Prepare Environment & Load Data #####

#load(file = "G:/My Drive/Walker Lab/SFD Spatial Dataset/Manuscript code.2/datafiles/.RData")

#load required packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(ggpubr)
  library(data.table)
  library(decontam)
  library(vegan)
  library(ggtext)
  library(outliers)
  library(rcompanion)
  library(Boruta)
  library(mgcv)
  library(lme4)
  library(glmmTMB)
  library(gratia)
  library(scales)
  library(janitor)
  library(caret)
  library(mikropml)
  library(gridExtra)
  library(classInt)
  library(phyloseq)
  library(ANCOMBC)
  library(RColorBrewer)
  library(pairwiseAdonis)
  library(emmeans)
  library(OptimalCutpoints)
  library(patchwork)
})

#prioritize dplyr functions
select <- dplyr::select
filter <- dplyr::filter

#set plotting options
theme_set(theme_classic() + 
            theme(panel.grid.major = element_blank(), 
                  panel.grid.minor = element_blank(),
                  panel.border = element_blank(),
                  axis.line = element_line(colour = "black"),
                  axis.text.y   = element_text(size=14),
                  axis.text.x   = element_text(size=14),
                  axis.title.y  = element_text(size=17, margin = margin(r = 10, l = 10)),
                  axis.title.x  = element_text(size=17, margin = margin(t = 10, b = 10)),
                  legend.text = element_text(size=11),
                  legend.title=element_text(size=13),
                  plot.title = element_text(size=21, hjust = 0.5),
                  legend.position = "top", 
                  legend.background = element_rect(fill="white", colour = NA),
                  plot.margin = margin(t = 10, r = 10))) 

#set working directory
if(Sys.info()[4] == "walkerlab-X9DR3-F"){
  setwd('/home/walkerlab/Documents/SFD spatial analyses/Manuscript code.2/datafiles')
} else {
  setwd("G:/My Drive/Walker lab/SFD Spatial Dataset/Manuscript code.2/datafiles/dry")
}

#get date for file saving 
date <- format(Sys.Date(), "%y%m%d")

#read in data
df <- fread("230213_SFDdatafull.formatted_ASR.csv", stringsAsFactors = T) %>% 
  mutate(oo_present = as.factor(oo_present),
         copies_rxn = replace_na(copies_rxn, 0),
         tn_ecoregion = na_if(tn_ecoregion, ''),
         log_copy_number = log10(copies_rxn+1))
tax <- fread('230428_SFDtaxfinal_engineer.csv')
seqmeta <- read.csv('221216_seqmeta.full_ASR.csv', stringsAsFactors = T) %>%
  mutate(swab_label = as.character(swab_label),
         swab_row = as.factor(str_replace_all(swab_well, "[0-9]", "")),
         swab_col = as.factor(str_replace_all(swab_well, "[A-Z]", "")),
         swab_plate_n = as.factor(as.numeric(swab_plate))) %>% 
  relocate(swab_row, swab_col, .after = swab_well)

#join sequencing metadata with dataframe 
df <- df %>%
  select(-DNA_con) %>% 
  left_join(seqmeta, by = "swab_label") %>%
  relocate(any_of(names(seqmeta)), .before = notes)

#extract otu matrix
otu.mat <- df %>%
  column_to_rownames("swab_label") %>% #assign swab names to row names
  select(contains("Otu")) 

#calculate alpha diversity metrics
df <- mutate(df,
             richness = rowSums(decostand(otu.mat, method = "pa")),
             shannon = diversity(otu.mat, index = "shannon"),
             evenness = exp(shannon)/(richness),
             simpson = diversity(otu.mat, index = "simpson"),
             invsimpson = diversity(otu.mat, index = "invsimpson"))

# create continuous time variables
df %>% 
  mutate(dayofyear = yday(date),
       year_f = as.factor(year),
       .after = year) -> df

#extract metadata only 
meta <- df %>% 
  select(-contains("Otu"))

#clean up working environment
keep.objs <- c(
  'date',
  'select',
  'filter',
  'df',
  'otu.mat',
  'tax',
  'meta',
  "even.gam.final",
  "rich.gam.final",
  'shan.gam.final',
  "alpha.multipanel")
remove.objs <- subset(ls(), !(ls() %in% keep.objs)) %>% 
  subset(. != "keep.objs")
rm(list = remove.objs) 

# Divide Pathogen Load to Generate disease state ####

log_copy_number.jenks <- subset(meta$log_copy_number, meta$log_copy_number > 0)
gvf <- c()
n.k <- c()
for(i in 1:14){
  k <- i+1
  gvf[i] <- jenks.tests(classIntervals(log_copy_number.jenks, k, style = "jenks"))[2]
  n.k[i] <- k
  if(i == 14){
    data.frame(goodness.of.fit = gvf,
               k.groups = n.k) %>% 
      ggplot(aes(y = goodness.of.fit, x = k.groups)) +
      geom_line(linewidth = 0.75) +
      geom_point(size = 3) +
      scale_y_continuous(limits = c(min(gvf), 1)) + 
      scale_x_continuous(breaks= pretty_breaks()) -> jenks.plot
  }
}
jenks.plot

lcn.splits <- classIntervals(log_copy_number.jenks, 3, style = "jenks")$brks
df <- mutate(df,
             dis.sev = factor(
               ifelse(log_copy_number == 0, "Neg",
                      ifelse(log_copy_number > 0 & log_copy_number <= lcn.splits[2], "Low",
                             ifelse(log_copy_number > lcn.splits[2] & log_copy_number <= lcn.splits[3], 'Moderate',
                                    ifelse(log_copy_number > lcn.splits[3], "Severe", NA)))),
               levels = c("Neg", "Low", "Moderate", "Severe")),
             dis.sev2 = fct_recode(dis.sev, `Low-moderate` = "Low", `Low-moderate` = "Moderate")
             
)
meta$dis.sev <- df$dis.sev
meta$dis.sev2 <- df$dis.sev2

#clean up working environment
keep.objs <- c(
  keep.objs,
  "lcn.splits"
)
remove.objs <- subset(ls(), !(ls() %in% keep.objs)) %>% 
  subset(. != "keep.objs")
rm(list = remove.objs)


# Modeling of Alpha Diversity ####

# OTU RICHNESS MODEL #

#check for outliers
grubbs.test(df$richness)
#no outliers

#examine distribution of response variable 
par(mar = c(1,1,1,1))
plotNormalDensity(df$richness, main = "Richness") #slight right skew
qqnorm(df$richness, main = "Richness") 
qqline(df$richness)
shapiro.test(df$richness)
#data is non-normal but approximates Gaussian distribution fairly well

#use boruta to select terms for rich gam
set.seed(123)
meta %>%
  select(-contains("lesion")) %>%
  select(-c(ct, tn_ecoregion, temperature_c, time,
            blood_smear, photos, sex, day, date,
            shannon, invsimpson, simpson, evenness, species,
            swab_label, swab_plate, blood_sample, copies_rxn, year,
            notes)) -> meta.rich
  drop_na(meta.rich) %>%
  Boruta(richness ~., data = ., doTrace = 2) -> rich.boruta
  plot(rich.boruta)
attStats(rich.boruta) %>%
  as.data.frame() %>%
  filter(decision != "Rejected") %>%
  arrange(desc(meanImp)) %>%
  rownames() -> rich.ft

# Set model formula for GAMM model of OTU Richness
rich.formula <- richness ~
  #parametric terms
  genus +
  ecomode +
  #univariate smooths
  s(log_copy_number) +
  s(dayofyear, by = oo_present) +
  s(DNA_con) +
  #Random effects
  s(site, bs = "re") +
  s(year_f, bs = "re") +
  s(swab_plate, bs = "re") +
  s(swab_col, bs = "re") +
  s(collector, bs = "re")


#Attempt to fit battery of distribution families
# rich.gam.1 <- gam(formula = rich.formula, family = gaussian(link = "identity"), data = meta, method = 'ML')
# rich.gam.2 <- gam(formula = rich.formula, family = gaussian(link = "log"), data = meta, method = 'ML')
# rich.gam.3 <- gam(formula = rich.formula, family = gaussian(link = "inverse"), data = meta, method = 'ML')
# rich.gam.4 <- gam(formula = rich.formula, family = poisson(link = "identity"), data = meta, method = 'ML')
# rich.gam.5 <- gam(formula = rich.formula, family = poisson(link = "identity"), data = meta, method = 'ML')
# rich.gam.6 <- gam(formula = rich.formula, family = poisson(link = "sqrt"), data = meta, method = 'ML')
# rich.gam.7 <- gam(formula = rich.formula, family = inverse.gaussian(link = "identity"), data = meta, method = 'ML')
# rich.gam.8 <- gam(formula = rich.formula, family = inverse.gaussian(link = "log"), data = meta, method = 'ML')
# rich.gam.9 <- gam(formula = rich.formula, family = inverse.gaussian(link = "inverse"), data = meta, method = 'ML')
# rich.gam.10 <- gam(formula = rich.formula, family = inverse.gaussian(link = "1/mu^2"), data = meta, method = 'ML')
# rich.gam.11 <- gam(formula = rich.formula, family = scat(link = "identity"), data = meta, method = 'ML')
# rich.gam.12 <- gam(formula = rich.formula, family = scat(link = "log"), data = meta, method = 'ML')
# rich.gam.13 <- gam(formula = rich.formula, family = scat(link = "inverse"), data = meta, method = 'ML')
# rich.gam.14 <- gam(formula = rich.formula, fameven.raw.figily = Gamma(link = "identity"), data = meta, method = 'ML')
# rich.gam.15 <- gam(formula = rich.formula, family = Gamma(link = "log"), data = meta, method = 'ML')
# rich.gam.16 <- gam(formula = rich.formula, family = Gamma(link = "inverse"), data = meta, method = 'ML')
# data.frame(mod = paste("rich.gam", 1:16, sep = "."),
#            aic = unlist(lapply(paste("rich.gam", 1:16, sep = "."), function(x) AIC(get(x))))) %>%
#   arrange(aic)

#Gammma distribution family preferred with log transformation link 
set.seed(123)
rich.gam.final <- gam(formula = rich.formula, family = Gamma(link = "log"), data = meta, method = 'REML')
#ensure model fit is appropriate 
appraise(rich.gam.final)
draw(rich.gam.final)
k.check(rich.gam.final)
#get final model results 
summary(rich.gam.final)

#generate figure of partial effect of pathogen load on richness 
smooth_estimates(rich.gam.final, 's(log_copy_number)') %>% 
  add_constant(coef(rich.gam.final)["(Intercept)"]) %>%
  add_confint() %>% 
  transform_fun(inv_link(rich.gam.final)) %>%
  mutate(copies_rxn = (10^log_copy_number)) %>%
  ggplot() +
  geom_ribbon(aes(x = copies_rxn, ymax = upper_ci, ymin = lower_ci), alpha = 0.15, color = "black") + 
  geom_line(aes(x = copies_rxn, y = est), linewidth = 0.75) +
  scale_x_log10(breaks = c(1, 11, 101, 1001, 10001),
                labels = c('0', '10', '100', '1000', '10000')) +
  annotate(geom = 'rect', ymin = -Inf, ymax = Inf, xmin = 0.9, xmax = 1.1, fill = "gray60", alpha = 0.3) +
  annotate(geom = 'rect', ymin = -Inf, ymax = Inf, xmin = 1.1, xmax = (10^lcn.splits[3])-1, fill = "darkorange", alpha = 0.3) +
  annotate(geom = 'rect', ymin = -Inf, ymax = Inf, xmin = (10^lcn.splits[3])-1, xmax = 10^lcn.splits[4], fill = "red2", alpha = 0.3) +
  geom_richtext(aes(x = 10, y = 400, label = "p < 0.001<br>R<sup>2</sup> = 0.294"), size = 6,
                label.padding = unit(c(0.6, 0.7, 0.35, 0.35), "lines"), label.colour = 'black') +
  labs(x = "Pathogen Load", 
       y = "Partial Effect") + 
  theme(plot.title = element_blank()) -> richgam.effect.fig

#generate figure of richness data with trendline 
ggplot(meta, aes(x = copies_rxn+1, y = richness)) + 
  geom_smooth(method = "gam", formula = y ~ s(x, k = 3), color = "black") +
  geom_point(data = . %>% filter(oo_present == 1), aes(fill = "Positive"), size = 3, shape = 21) +
  geom_point(data = . %>% filter(oo_present == 0), color = "#377EB8", alpha = 0.2, size = 3, show.legend = F) +
  geom_boxplot(data = df %>% filter(oo_present == 0), aes(fill = "BDL"), width = 0.15, show.legend = F, linewidth = 0.75) + 
  scale_fill_manual(values = c("#377EB8", "#E41A1C")) +
  scale_x_log10(breaks = c(1, 11, 101, 1001, 10001),
                labels = c('0', '10', '100', '1000', '10000')) +
  labs(x = "Pathogen Load", y = "OTU Richness", fill = "qPCR") +
  theme(legend.position = 'none') -> rich.raw.fig

# SHANNON DIVERSITY MODEL #

#model Shannon diversity using GAM methodology
#Set model formula for GAMM model of OTU Richness
shan.formula <- shannon ~ 
  #parametric terms 
  genus + 
  ecomode +
  #univariate smooths
  s(log_copy_number) +
  s(dayofyear, by = oo_present) +
  s(DNA_con) +
  #Random effects 
  s(site, bs = "re") +
  s(year_f, bs = "re") + 
  s(swab_plate, bs = "re") +
  s(swab_col, bs = "re") +
  s(collector, bs = "re") 
#Attempt to fit battery of distribution families 
# shan.gam.1 <- gam(formula = shan.formula, family = gaussian(link = "identity"), data = meta, method = 'ML')
# shan.gam.2 <- gam(formula = shan.formula, family = gaussian(link = "log"), data = meta, method = 'ML')
# shan.gam.3 <- gam(formula = shan.formula, family = gaussian(link = "inverse"), data = meta, method = 'ML')
# shan.gam.4 <- gam(formula = shan.formula, family = inverse.gaussian(link = "identity"), data = meta, method = 'ML')
# shan.gam.5 <- gam(formula = shan.formula, family = inverse.gaussian(link = "log"), data = meta, method = 'ML')
# shan.gam.6 <- gam(formula = shan.formula, family = inverse.gaussian(link = "inverse"), data = meta, method = 'ML')
# shan.gam.7 <- gam(formula = shan.formula, family = inverse.gaussian(link = "1/mu^2"), data = meta, method = 'ML')
# shan.gam.8 <- gam(formula = shan.formula, family = scat(link = "identity"), data = meta, method = 'ML')
# shan.gam.9 <- gam(formula = shan.formula, family = scat(link = "log"), data = meta, method = 'ML')
# shan.gam.10 <- gam(formula = shan.formula, family = scat(link = "inverse"), data = meta, method = 'ML')
# shan.gam.11 <- gam(formula = shan.formula, family = Gamma(link = "identity"), data = meta, method = 'ML')
# shan.gam.12 <- gam(formula = shan.formula, family = Gamma(link = "log"), data = meta, method = 'ML')
# shan.gam.13 <- gam(formula = shan.formula, family = Gamma(link = "inverse"), data = meta, method = 'ML')
# data.frame(mod = paste("shan.gam", 1:13, sep = "."),
#            aic = unlist(lapply(paste("shan.gam", 1:13, sep = "."), function(x) AIC(get(x))))) %>% 
#   arrange(aic)

#Gaussian distribution family preferred with no transformation link
set.seed(123)
shan.gam.final <- gam(formula = shan.formula, family = Gamma(link = "log"), data = meta, method = 'REML')
#ensure model fit is appropriate 
appraise(shan.gam.final)
draw(shan.gam.final)
k.check(shan.gam.final)
#get final model results 
summary(shan.gam.final)

#generate figure of partial effect of pathogen load on Shannon diversity 
smooth_estimates(shan.gam.final, 's(log_copy_number)') %>% 
  add_constant(coef(shan.gam.final)["(Intercept)"]) %>%
  add_confint() %>% 
  transform_fun(inv_link(shan.gam.final)) %>% 
  mutate(copies_rxn = (10^log_copy_number)) %>%
  ggplot() +
  geom_ribbon(aes(x = copies_rxn, ymax = upper_ci, ymin = lower_ci), alpha = 0.15, color = "black") + 
  geom_line(aes(x = copies_rxn, y = est), linewidth = 0.75) +
  scale_x_log10(breaks = c(1, 11, 101, 1001, 10001),
                labels = c('0', '10', '100', '1000', '10000'),
                limits = c(0.9,29349)) +
  scale_y_continuous(limits = c(2,5),
                     breaks = c(2.25, 3.5, 4.75)) +
  annotate(geom = 'rect', ymin = -Inf, ymax = Inf, xmin = 0.9, xmax = 1.1, fill = "gray60", alpha = 0.3) +
  annotate(geom = 'rect', ymin = -Inf, ymax = Inf, xmin = 1.1, xmax = (10^lcn.splits[3])-1, fill = "darkorange", alpha = 0.3) +
  annotate(geom = 'rect', ymin = -Inf, ymax = Inf, xmin = (10^lcn.splits[3])-1, xmax = 10^lcn.splits[4], fill = "red2", alpha = 0.3) +
  geom_richtext(aes(x = 10, y = 2.35, label = "p < 0.001<br>R<sup>2</sup> = 0.208"), size = 6,
                label.padding = unit(c(0.6, 0.7, 0.35, 0.35), "lines"), label.colour = 'black') +
  labs(x = "Pathogen Load", 
       y = "Partial Effect") + 
  theme(plot.title = element_blank()) -> shangam.effect.fig


#generate figure of shannon diversity data with trendline 
ggplot(meta, aes(x = copies_rxn+1, y = shannon)) + 
  geom_smooth(method = "gam", formula = y ~ s(x, k = 3), color = "black") +
  geom_point(data = . %>% filter(oo_present == 1), aes(fill = "Positive"), size = 3, shape = 21) +
  geom_point(data = . %>% filter(oo_present == 0), color = "#377EB8", alpha = 0.2, size = 3, show.legend = F) +
  geom_boxplot(data = df %>% filter(oo_present == 0), aes(fill = "BDL"), width = 0.15, show.legend = F, linewidth = 0.75) + 
  scale_fill_manual(values = c("#377EB8", "#E41A1C")) +
  scale_x_log10(breaks = c(1, 11, 101, 1001, 10001),
                labels = c('0', '10', '100', '1000', '10000')) +
  labs(x = "Pathogen Load", y = "Shannon Diversity", fill = "qPCR") +
  theme(legend.position = 'none') -> shan.raw.fig

#clean up working environment
keep.objs <- c(
  keep.objs,
  "shan.gam.final",
  "rich.gam.final",
  'rich.raw.fig',
  'richgam.effect.fig',
  'shangam.effect.fig',
  'shan.raw.fig')
remove.objs <- subset(ls(), !(ls() %in% keep.objs)) %>%
  subset(. != "keep.objs")
rm(list = remove.objs)

# Modeling of Community Composition #### 

#subset samples to remove under sampled genera and county 
meta %>% 
  group_by(genus) %>% 
  filter(n() > 10) %>% 
  ungroup() %>% 
  filter(state == "TN") %>% 
  group_by(county) %>% 
  filter(n() > 10) %>% 
  ungroup() -> beta.meta
#this is necessary to ensure multivariate dispersion is not artificially over inflated 

#subset otu matrix accordingly 
beta.otu.mat <- otu.mat[rownames(otu.mat) %in% beta.meta$swab_label,]

#create distances matrices 
#Bray-Curtis
bray <- vegdist(beta.otu.mat, method="bray") 
#Raup-Crick
# raup <- raupcrick(beta.otu.mat,  null = 'r1', nsimul = 999) 
# saveRDS(raup, file = "raupdis.rds")
raup <- readRDS(file = "raupdis.rds")

#
#betadisper to determine homogeneity of groups by disease state
#

#Bray-Curtis
bray.betadisper.ds <- betadisper(bray, beta.meta$dis.sev2)
permutest(bray.betadisper.ds) # significant difference in multivariate dispersion
TukeyHSD(bray.betadisper.ds) # difference between negative and moderate infections only 
#visualize this result 
par(mfrow=c(1,2))
plot(bray.betadisper.ds, label = T, label.cex = 0.5, col = c("gray", 'darkorange', "red2"))
boxplot(bray.betadisper.ds, col=  c("gray", 'darkorange', "red2"))
par(mfrow=c(1,1))
# PERMANOVA is currently overly conservative (larger group has greater dispersion)
# see http://doi.org/10.1890/12-2010.1

#Raup-Crick
raup.betadisper.ds <- betadisper(raup, beta.meta$dis.sev2)
permutest(raup.betadisper.ds) # No significant difference in multivariate dispersion
TukeyHSD(raup.betadisper.ds) # difference between low and moderate infections only 


#
#Impliment PERMANOVA to determine if centroid position varies by pathogen load and richness
#

#Bray-Curtis
set.seed(123) #set seed for reproducibility
bray.perm <- adonis2(bray ~ dis.sev2 + richness + richness:oo_present, strata = beta.meta$site, data=beta.meta, nperm = 999)
bray.perm

#Raup-Crick
set.seed(123) #set seed for reproducibility
raup.perm <- adonis2(raup ~ dis.sev2 + richness + richness:oo_present, strata = beta.meta$county, data=beta.meta, nperm = 999)
raup.perm

#
#utilize pairwise Adonis to determine which disease states are associated with different communities 
#

#bray 
pairwise.adonis(bray, beta.meta$dis.sev2)
# all disease states are associated 
# with different assemblages

#raup-crick
pairwise.adonis(raup, beta.meta$dis.sev2)
# only Low-Moderate and Negative community are different when
# explicitly accounting for richness variation 

#
#visualize effect of disease state with NMDS 
#

#plot NDMS for Bray-Curtis 
set.seed(123) #set seed for reproducibility 
ds.nmds<-metaMDS(bray, k=3) #conduct Non-metric dimensional scaling  
df.ds.ord <- as.data.frame(ds.nmds[["points"]]) %>% #generate dataframe with NMDS points 
  rownames_to_column("swab_label") %>% 
  left_join(beta.meta)
#render and save plot 
df.ds.ord %>% 
  mutate(dis.sev2 = ifelse(dis.sev %in% c("Low", "Moderate"), "Low-Moderate",
                           ifelse(dis.sev == "Neg", "Negative", as.character(dis.sev))),
         dis.sev2 = factor(dis.sev2, levels = c("Negative", "Low-Moderate", "Severe"))) %>% 
  ggplot(aes(x = MDS1, y = MDS2)) + 
  stat_ellipse(aes(color = dis.sev2), linewidth = 1, linetype = "longdash") + 
  geom_point(aes(color = dis.sev2), size = 2.5, alpha = 0.5) + 
  stat_ellipse(level = 1e-10, geom = "point", aes(fill = dis.sev2), size = 7, shape = 21) + 
  scale_color_manual(values = c("gray60", 'darkorange', "red2")) + 
  scale_fill_manual(values = c("gray60", 'darkorange', "red2")) + 
  annotate(geom = "text", x = max(df.ds.ord$MDS1)*0.95, y = max(df.ds.ord$MDS2)*0.9, label = "Bray-Curtis\np = 0.001", size = 4, hjust = 1) +
  labs(fill = "Disease State",
       color = "Disease State",
       title = "Bray-Curtis") +
  theme(legend.position = 'bottom') -> bray.nmds

#plot NDMS for Raup-Crick 
set.seed(123) #set seed for reproducibility 
ds.nmds<-metaMDS(raup, k=3) #conduct Non-metric dimensional scaling  
df.ds.ord <- as.data.frame(ds.nmds[["points"]]) %>% #generate dataframe with NMDS points 
  rownames_to_column("swab_label") %>% 
  left_join(beta.meta)
#render and save plot 
df.ds.ord %>% 
  mutate(dis.sev2 = ifelse(dis.sev %in% c("Low", "Moderate"), "Low-Moderate",
                           ifelse(dis.sev == "Neg", "Negative", as.character(dis.sev))),
         dis.sev2 = factor(dis.sev2, levels = c("Negative", "Low-Moderate", "Severe"))) %>% 
  ggplot(aes(x = MDS1, y = MDS2)) + 
  stat_ellipse(aes(color = dis.sev2), linewidth = 1, linetype = "longdash") + 
  geom_point(aes(color = dis.sev2), size = 2.5, alpha = 0.5) + 
  stat_ellipse(level = 1e-10, geom = "point", aes(fill = dis.sev2), size = 7, shape = 21) + 
  scale_color_manual(values = c("gray60", 'darkorange', "red2")) + 
  scale_fill_manual(values = c("gray60", 'darkorange', "red2")) + 
  annotate(geom = "text", x = max(df.ds.ord$MDS1)*0.95, y = max(df.ds.ord$MDS2)*0.9, label = "Raup-Crick\np = 0.004", size = 4, hjust = 1) +
  labs(fill = "Disease State",
       color = "Disease State",
       title = "Raup-Crick") +
  theme(legend.position = 'bottom') -> raup.nmds

#
#Compare effect of richness vs pathogen load with dbRDA
#

#bray curtis 
cap.bray <- capscale(bray ~ richness + log_copy_number, data = beta.meta)
df.cap.bray <- data.frame(summary(cap.bray)$sites[,1:2]) %>% 
  rownames_to_column('swab_label') %>% 
  left_join(beta.meta) 

#raup-crick
cap.raup <- capscale(raup ~ richness + log_copy_number, data = beta.meta)
df.cap.raup <- data.frame(summary(cap.raup)$sites[,1:2]) %>% 
  rownames_to_column('swab_label') %>% 
  left_join(beta.meta) 

#
#Plot dbRDA
#

#bray
cap.bray.pe <- as.character(round((summary(cap.bray)$concont$importance[2,]*summary(cap.bray)$constr.chi) / summary(cap.bray)$tot.chi * 100, 2))
df.cap.bray %>%
  mutate(dis.sev2 = ifelse(dis.sev %in% c("Low", "Moderate"), "Low-Moderate",
                           ifelse(dis.sev == "Neg", "Negative", as.character(dis.sev))),
         dis.sev2 = factor(dis.sev2, levels = c("Negative", "Low-Moderate", "Severe"))) %>%
  ggplot(aes(x=CAP1, y=CAP2, color = dis.sev2)) +
  geom_point(size = 2, alpha = 0.5) + 
  stat_ellipse(linewidth = 1, linetype = "longdash") + 
  annotate(geom = "text", x=cap.bray$CCA$biplot[2,1]-.1, y=cap.bray$CCA$biplot[2,2]*1.52, label = "Pathogen\nLoad", color = "black", size = 4, fontface = "bold") + 
  annotate(geom = "text", x=cap.bray$CCA$biplot[1,1]*1.55, y=cap.bray$CCA$biplot[1,2]+.025, label = "Richness", color = "black", size = 4, fontface = "bold") +
  annotate(geom = "text", x=max(df.cap.bray$CAP1)*0.8, y=5.25, label = "Bray-Curtis", color = "black", size = 4.25) +
  stat_ellipse(level = 1e-10, geom = "point", aes(fill = dis.sev2), size = 7, shape = 21, color = "black") + 
  geom_segment(aes(x = 0, y = 0, xend = cap.bray$CCA$biplot[1,1], yend = cap.bray$CCA$biplot[1,2]),
               linewidth = 0.75,
               color = "gray20",
               arrow = arrow(length = unit(0.5, "cm"))) +
  geom_segment(aes(x = 0, y = 0, xend = cap.bray$CCA$biplot[2,1], yend = cap.bray$CCA$biplot[2,2]),
               linewidth = 0.75,
               color = "gray20",
               arrow = arrow(length = unit(0.5, "cm"))) +
  labs(fill = "Disease State",
       color = "Disease State",
       title = "Bray-Curtis",
       x = paste0("CAP1 (", cap.bray.pe[1], "%)"),
       y = paste0("CAP2 (", cap.bray.pe[2], "%)")) +
  scale_color_manual(values = c("gray60", 'darkorange', "red2")) + 
  scale_fill_manual(values = c("gray60", 'darkorange', "red2")) + 
  theme(legend.position = c(0.125,0.165),
        plot.margin = margin(10,10,10,10),
        plot.title = element_blank()) -> bray.cap

#
# Examine community structure at disease state multivariate centroid positions 
#

#calculate multivariate centroid positions for each disease state category
cbind(
  data.frame('V1' = c("cent.Neg", "cent.Low-Moderate", "cent.Severe")),
  as.data.frame(
    rbind(
      round(colMeans(beta.otu.mat[beta.meta$dis.sev == "Neg",])),
      round(colMeans(beta.otu.mat[beta.meta$dis.sev %in% c("Low", "Moderate"),])),
      round(colMeans(beta.otu.mat[beta.meta$dis.sev == "Severe",]))
    )
  )
) %>% 
  column_to_rownames('V1') %>% 
  rbind(., beta.otu.mat) %>% 
  as.matrix() -> cent.mat 

#generate new metadata object that accommodates centroid "samples"
beta.meta %>% 
  ungroup() %>%
  full_join(.,
            data.frame(swab_label = c("cent.Neg", "cent.Low-Moderate", "cent.Severe"),
                       dis.sev = c("Negative", "Low-Moderate", "Severe"))
  ) %>% 
  mutate(dis.sev = ifelse(dis.sev %in% c("Low", "Moderate"), "Low-Moderate",
                          ifelse(dis.sev == "Neg", "Negative", as.character(dis.sev))),
         dis.sev = factor(dis.sev, levels = c("Negative", "Low-Moderate", "Severe")),
         centroid = ifelse(grepl("cent.", swab_label), TRUE, FALSE)) -> cent.beta.meta

# generate a phyloseq object for assembly composition visualization
# format OTU abundance table
otu.table <- otu_table(cent.mat[grepl("cent.", rownames(cent.mat)),], taxa_are_rows = F)
# format sample metadata
sample.data <-sample_data(column_to_rownames(cent.beta.meta[cent.beta.meta$centroid,], var = "swab_label"))
# format taxonomy table 
tax.table <- tax %>%
  filter(otu %in% taxa_names(otu.table)) %>%
  column_to_rownames(var="otu") %>%
  mutate(genus = case_when(grepl('unclassified', tax$genus) ~ paste("Unclassified", tax$genus),
                           !grepl('unclassified', tax$genus) ~ tax$genus),
         genus = str_replace(genus, "_unclassified", "")) %>% 
  as.matrix() %>%
  tax_table()
# merge formatted dataframes into phyloseq object
cent.phylo <- merge_phyloseq(otu.table, sample.data, tax.table)
gc() # clear memory extraneous usage 

#nest taxonomy data to allow for efficient visualization
cent.phylo.nest <- fantaxtic::nested_top_taxa(cent.phylo,
                                              top_tax_level = "phylum",
                                              nested_tax_level = "genus",
                                              n_top_taxa = 5, 
                                              n_nested_taxa = 3)

#generate stacked bar plot of assemblage composition at each disease state centroid position 
fantaxtic::plot_nested_bar(ps_obj = cent.phylo.nest$ps_obj,
                           top_level = "phylum",
                           nested_level = "genus",
                           sample_order = c("cent.Neg", "cent.Low-Moderate", "cent.Severe")
) +
  scale_y_continuous(labels = scales::percent) + 
  scale_x_discrete(labels = c("<span style = 'color:gray60;'>Negative</span>",
                              "<span style = 'color:darkorange;'>Low-moderate</span>",
                              "<span style = 'color:red2;'>Severe</span>")) +
  theme_classic() + 
  theme(legend.text = element_markdown(size = 9), 
        axis.title = element_text(size = 15),
        axis.title.x = element_text(margin = margin(10,0,5,0)),
        axis.title.y = element_text(margin = margin(0,10,0,10)),
        axis.text.x = element_markdown(size = 12, color = "black"),
        axis.text.y = element_text(size = 12, color = "black"),
        plot.margin = margin(10,10,10,10)) + 
  labs(x = "Disease State",
       y = "Relative Abundance") + 
  guides(fill=guide_legend(ncol=1, title = "Genus"),
         color=guide_legend(ncol=1, title = "Genus")) -> cent.bp

#slightly reformat plots for multipanel 
bray.nmds.f <- bray.nmds + theme(plot.title = element_blank(),
                                 legend.position = "none")
raup.nmds.f <- raup.nmds + theme(plot.title = element_blank(),
                                 legend.position = "none")
bray.cap.f <- bray.cap + theme(plot.title = element_blank(),
                               legend.position = "none") +
  scale_y_continuous(limits = c(-4.5, 5.25))

#use 'patchwork' package to arrange plots into multipanel 
bray.nmds.f + raup.nmds.f + bray.cap.f + get_legend(bray.nmds) + cent.bp + plot_spacer() +
  plot_layout(heights = c(0.5, 0.5, 0.001, 1.35),
              widths = c(1,0.95, 0.05),
              design = "
              133
              233
              444
              556
              ") + 
  plot_annotation(tag_levels = list(c("A", "B", "C", "", "D"))) &
  theme(plot.tag = element_text(size = 20,
                                face = "bold",
                                color = "black",
                                family = NULL,
                                hjust = 10)) -> comp.multi

#save high resolution image file of multipanel figure
ggsave(plot = comp.multi,
       filename = "comp.multi.jpg",
       path = str_replace(getwd(), "datafiles", "Figures"),
       device = "jpg",
       height = 5000,
       width = 4000,
       units = 'px')

#clean up working environment
keep.objs <- c(
  keep.objs,
  'beta.meta',
  'cent.phylo',
  'comp.multi',
  "bray",
  "raup")
remove.objs <- subset(ls(), !(ls() %in% keep.objs)) %>%
  subset(. != "keep.objs")
rm(list = remove.objs)


# Modeling of Distance-to-Centroid ####

#
#generate distance to centroid values 
#

#Bray-Curtis
bray.d2c <- betadisper(bray, beta.meta$ecomode)

#Raup-Crick
raup.d2c <- {options(warn = 1) #silence warnings about zeros in raup crick distance to centroid matrix
  betadisper(raup, beta.meta$ecomode)}

#join with metadata
cbind(beta.meta,
      bray.dis = bray.d2c$distances,
      raup.dis = raup.d2c$distances) %>% 
  mutate(dis.sev2 = ifelse(dis.sev %in% c("Low", "Moderate"), "Low-Moderate",
                           ifelse(dis.sev == "Neg", "Negative", as.character(dis.sev))),
         dis.sev2 = factor(dis.sev2, levels = c("Negative", "Low-Moderate", "Severe"))) -> beta.meta.dis

#bray-curtis distance to centroid model
bray.dis.m <- glmmTMB(bray.dis ~ dis.sev2 + (1|site) + (1|swab_plate),
                      data  = beta.meta.dis,
                      family  = ordbeta(link = "logit")) 
#check model fit
plot(DHARMa::simulateResiduals(bray.dis.m))
#get model results 
car::Anova(bray.dis.m, type = "II")
#perform post-hoc pairwise comparisons 
contrast(emmeans(bray.dis.m, "dis.sev2"), "pairwise")
#get compact letter display to show significant differences 
multcomp::cld(object = emmeans(bray.dis.m, "dis.sev2"),
              Letters = letters,
              reverse = T,
              alpha = 0.05)

#raup-curtis distance to centroid model
raup.dis.m <- glmmTMB(raup.dis ~ dis.sev2 + (1|site) + (1|swab_plate),
                      data  = beta.meta.dis,
                      family  = ordbeta(link = "logit")) 
#check model fit
plot(DHARMa::simulateResiduals(raup.dis.m))
#get model results 
summary(raup.dis.m)
car::Anova(raup.dis.m, type = "II")

#generate cld data 
data.frame(
  dis.sev2 = levels(beta.meta.dis$dis.sev2),
  value = c(.805, .75, .795),
  cld = c('b', 'b', 'a')
) -> dat.cld

data.frame(
  dis.sev2 = "Low-Moderate",
  value = 0.75,
  lab = "No Significant Differences"
) -> dat.an

beta.meta.dis %>% 
  ggplot(aes(y = raup.dis, x = dis.sev2, fill = dis.sev2)) + 
  geom_violin() +
  geom_boxplot(fill = "white", alpha = 0.5, width = 0.125, outlier.alpha = 1, outlier.size = 2) + 
  geom_text(data = dat.an, aes(y = value, x = dis.sev2, label = lab), hjust = 0.5, size = 4) +
  scale_fill_manual(values = c("gray60", 'darkorange', "red2")) + 
  scale_y_continuous(limits = c(0,0.875), name = "Distance-to-Centroid") +
  geom_text(aes(x = 0.5, y  = 0.875, label = "Raup-Crick"),hjust = 0, size = 5, check_overlap = TRUE) +
  theme(axis.title.x = element_blank(),
        legend.position = "none",
        plot.margin = margin(10,10,10,10),
        strip.text = element_text(size = 15)) -> d2c.raup

beta.meta.dis %>% 
  ggplot(aes(y = bray.dis, x = dis.sev2, fill = dis.sev2)) + 
  geom_violin() +
  geom_boxplot(fill = "white", alpha = 0.5, width = 0.125, outlier.alpha = 1, outlier.size = 2) + 
  geom_text(data = dat.cld, aes(y = value, x = dis.sev2, label = cld), hjust = 0.5, size = 7) +
  scale_fill_manual(values = c("gray60", 'darkorange', "red2")) + 
  scale_y_continuous(limits = c(min(beta.meta.dis$bray.dis),0.85), name = "Distance-to-Centroid") +
  geom_text(aes(x = 0.5, y  = 0.85, label = "Bray-Curtis"),hjust = 0, size = 5, check_overlap = TRUE) +
  theme(axis.title.x = element_blank(),
        legend.position = "none",
        plot.margin = margin(10,10,10,10),
        strip.text = element_text(size = 15)) -> d2c.bray

ggarrange(d2c.bray, d2c.raup,
          ncol = 2, 
          labels = c("E", "F"),
          heights = c(2.2, 1),
          font.label = list(size = 20)) -> d2c.fig

#generate alpha and beta diversity multipanel 
ggarrange(rich.raw.fig,
          richgam.effect.fig,
          shan.raw.fig,
          shangam.effect.fig,
          ncol = 2,
          nrow = 2,
          legend = "none",
          labels = LETTERS[1:4],
          font.label = list(size = 20), 
          align = 'hv') %>%
  ggarrange(
    .,
    d2c.fig,
    nrow = 2,
    heights = c(1.5, 1)
  ) -> diversity.multi

ggsave(plot = diversity.multi,
       filename = '231012_diversity.multi.jpg',
       device = 'jpg',
       units = 'px',
      ##width = 3000,
       dpi = 320,
       path = str_replace(getwd(), "datafiles", "Figures"))

#clean up working environment
keep.objs <- c(
  keep.objs,
  'diversity.multi',
  'raup.dis.m',
  'bray.dis.m')
remove.objs <- subset(ls(), !(ls() %in% keep.objs)) %>%
  subset(. != "keep.objs") %>% 
  c(., 'rich.raw.fig', 'richgam.effect.fig', 'shan.raw.fig', 'shangam.effect.fig')
rm(list = remove.objs)

# ML to ID Taxa Associated with Richness & Pathogen Load ####

#prepare environment for parallel processing
doFuture::registerDoFuture()
n.cores <- floor(parallel::detectCores()-1)
future::plan(future::multisession, workers = n.cores)
options(doFuture.rng.onMisuse = "ignore") #Silence false positive warning about random number generation

#generate genus abundance matrix 
# otu.mat %>% 
#   rownames_to_column("sample.id") %>% 
#   pivot_longer(-sample.id, names_to = 'otu') %>% 
#   left_join(tax) %>%
#   select(-subset(names(tax), !names(tax) %in% c("otu", "genus"))) %>% 
#   group_by(sample.id, genus) %>% 
#   mutate(value = sum(value),
#          .keep = "used") %>% 
#   sample_n(1) %>% 
#   ungroup() %>% 
#   pivot_wider(names_from = 'genus') %>%
#   column_to_rownames('sample.id')  %>% 
#   select(names(sort(colSums(.), decreasing = TRUE))) %>% 
#   .[order(match(rownames(.),rownames(otu.mat))),] -> genus.mat 
# saveRDS(genus.mat, file = 'genus.mat.rds')
genus.mat <- readRDS(file = 'genus.mat.rds')

#
## Richness Model
#

#generate dataframe for richness model 
left_join(select(meta, swab_label, oo_present, richness, log_copy_number),
          rownames_to_column(genus.mat, 'swab_label')) %>% 
  column_to_rownames("swab_label") %>% 
  select(richness, any_of(names(genus.mat))) -> rich_mod.df

# run GLMnet model in mikropml
# rich_mod <- run_ml(
#   #specify dataset, algorithim, and response 
#   dataset = rich_mod.df,
#   method = "glmnet",
#   outcome_colname = "richness",
#   # use Leave-one-out Cross-Validation for model training
#   cross_vall = trainControl(
#     method = 'LOOCV',                # k-fold cross validation 'cv'
#     number = 1,                      # number of folds
#     savePredictions = 'final',       # saves predictions for optimal tuning parameter
#     classProbs = F                   # should class probabilities be returned
#   ),
#   # use grid search to conduct hyper-parameter selection
#   hyperparameters = expand.grid(alpha = seq(0, 1, by = 0.1),
#                                 lambda = seq(0, 1, length.out = 11)),
#   #set R-squared as parameter to evaluate model performance
#   perf_metric_name = "RMSE",
#   perf_metric_function = defaultSummary,
#   #Center and scale features
#   preProcess = c('center', 'scale'),
#   #set random seed
#   seed = 12345
# )
# saveRDS(rich_mod, file = 'rich_mod.rds')
rich_mod <- readRDS(file = 'rich_mod.rds')

#examine overall model performance 
rich_mod$performance

#Visualize observed vs predicted values 
data.frame(predicted = predict(rich_mod$trained_model, select(rich_mod$test_data, -richness)),
           observed = pull(rich_mod$test_data, richness)) %>% 
  ggplot(aes(observed, predicted)) + 
  geom_point() + 
  geom_smooth(method = "lm") + 
  ggtitle("Richness GLMnet")
  
#examine effect of hyper parameter tuning on model performance 
annotate_figure( 
  ggpubr::ggarrange(
    plot_hp_performance(get_hp_performance(rich_mod$trained_model)$dat, lambda, RMSE),
    plot_hp_performance(get_hp_performance(rich_mod$trained_model)$dat, alpha, RMSE)
  ),
  top = text_grob("Richness GlMnet", size = 14)
)

#perform feature importance on richness model 
# rich_fi <- get_feature_importance(
#   #specify trained model, test data, and outcome 
#   trained_model = rich_mod$trained_model,
#   test_data = rich_mod$test_data,
#   outcome_colname = 'richness',
#   #set R-squared as parameter to evaluate model performance
#   perf_metric_function = defaultSummary,
#   perf_metric_name = "RMSE",
#   class_probs = F,
#   #specify algorithim used to train model
#   method = "glmnet",
#   #seed random seed
#   seed = 12345)
# saveRDS(rich_fi, file = 'rich_fi.rds')
rich_fi <- readRDS(file = 'rich_fi.rds')

# get dataframe of significant genera 
rich_fi %>% 
  relocate(genus = feat) %>% 
  left_join(distinct(select(tax, -otu))) %>% 
  filter(pvalue < .05) %>% 
  arrange(desc(perf_metric_diff)) %>%
  distinct(genus, .keep_all = T) %>% 
  relocate(any_of(names(tax))) %>% 
  mutate(outcome = "richness") %>% 
  mutate(across(c(perf_metric, perf_metric_diff), ~./mean(meta$richness))) -> df.rich.fi

#
## Pathogen Load Model
#

#generate dataframe for pathogen load model 
left_join(select(meta, swab_label, oo_present, richness, log_copy_number),
          rownames_to_column(genus.mat, 'swab_label')) %>% 
  column_to_rownames("swab_label") %>% 
  select(log_copy_number, any_of(names(genus.mat))) -> pl_mod.df

# run Random Forest  model in mikropml
# pl_mod <- run_ml(
#   #specify dataset, algorithim, and response 
#   dataset = pl_mod.df,
#   method = "rf",
#   outcome_colname = "log_copy_number",
#   # use Leave-one-out Cross-Validation for model training
#   cross_vall = trainControl(
#     method = 'LOOCV',                # k-fold cross validation 'cv'
#     number = 1,                      # number of folds
#     savePredictions = 'final',       # saves predictions for optimal tuning parameter
#     classProbs = F                   # should class probabilities be returned
#   ),
#   # use grid search to conduct hyper-parameter selection
#   hyperparameters = data.frame(mtry = c(1:15)),
#   #set R-squared as parameter to evaluate model performance
#   perf_metric_name = "RMSE",
#   perf_metric_function = defaultSummary,
#   #set random seed
#   seed = 12345
# )
# saveRDS(pl_mod, file = 'pl_mod.rds')
pl_mod <- readRDS(file = 'pl_mod.rds')

#examine overall model performance 
pl_mod$performance

#Visualize observed vs predicted values 
data.frame(predicted = predict(pl_mod$trained_model, select(pl_mod$test_data, -log_copy_number)),
           observed = pull(pl_mod$test_data, log_copy_number)) %>% 
  ggplot(aes(observed, predicted)) + 
  geom_point() + 
  geom_smooth(method = "lm") + 
  ggtitle("Pathogen Load Random Forest")

#examine effect of hyper-parameter tuning on model performance 
plot_hp_performance(get_hp_performance(pl_mod$trained_model)$dat, mtry, RMSE)

#perform feature importance on pathogen load model 
# pl_fi <- get_feature_importance(
#   #specify trained model, test data, and outcome 
#   trained_model = pl_mod$trained_model,
#   test_data = pl_mod$test_data,
#   outcome_colname = 'log_copy_number',
#   #set R-squared as parameter to evaluate model performance
#   perf_metric_function = defaultSummary,
#   perf_metric_name = "RMSE",
#   class_probs = F,
#   #specify algorithim used to train model
#   method = "glmnet",
#   #seed random seed
#   seed = 12345)
# saveRDS(pl_fi, file = 'pl_fi.rds')
pl_fi <- readRDS(file = 'pl_fi.rds')

# get dataframe of significant genera 
pl_fi %>% 
  relocate(genus = feat) %>% 
  left_join(distinct(select(tax, -otu))) %>% 
  filter(pvalue < .05) %>% 
  arrange(desc(perf_metric_diff)) %>%
  distinct(genus, .keep_all = T) %>% 
  relocate(any_of(names(tax))) %>% 
  mutate(outcome = "log_copy_number") %>% 
  mutate(across(c(perf_metric, perf_metric_diff), ~./mean(meta$log_copy_number))) -> df.pl.fi

#stop cluster 
future:::ClusterRegistry("stop")

#
# Generate summary plot 
#

#generate simplified legend for plot
get_legend(
  ggplot(data = data.frame(outcome = fct_inorder(rev(c("Richness", 'Pathogen Load'))),
                           y = c(1,1)),
         aes(x = outcome, y = y, fill = outcome)) +
    geom_col(color = "black") +
    scale_fill_manual(values = c("gray50", "gray90")) +
    labs(fill = "Outcome")  +
    theme(legend.position = "right",
          legend.text = element_text(size = 12),
          legend.title = element_text(size = 15))
) %>% 
  as_ggplot() %>% 
  ggplotGrob() -> phylum.plot.legend

#generate feature importance plot by phylum
rbind(df.rich.fi,
      df.pl.fi) %>% 
  group_by(outcome, phylum) %>% 
  summarize(perf_metric_diff = sum(perf_metric_diff)) %>% 
  ungroup() %>% 
  pivot_wider(names_from = "outcome", values_from = "perf_metric_diff") %>% 
  rowwise() %>% 
  mutate(perf_metric_diff_sum = sum(log_copy_number, richness, na.rm = T)) %>% 
  ungroup() %>% 
  arrange(desc(perf_metric_diff_sum)) %>% 
  slice_head(n = 5) %>% 
  select(c(1:3)) %>% 
  pivot_longer(-phylum, values_to = 'perf_metric_diff', names_to = "outcome") %>% 
  mutate(phylum = fct_reorder(phylum, perf_metric_diff, .desc = T, .fun = sum),
         phylum2 = ifelse(phylum %in% c("Proteobacteria", "Bacteroidetes", "Actinobacteria", "Firmicutes"), as.character(phylum), "Other"),
         outcome = fct_rev(outcome)) %>% 
  left_join(data.frame(
    expand.grid(
      phylum2 = c("Proteobacteria", "Bacteroidetes", "Actinobacteria", "Firmicutes", "Other"),
      outcome = c("richness", "log_copy_number")),
    col = c("#ff8a9f", "#fdff8a", "#8aceff", "#8affa3", 'gray90', 
            "#a32d43", "#a1a32d", '#2d72a3', "#2da346", 'gray50')),
    by = c("phylum2", "outcome")
  ) %>% 
  ggplot(aes(x = phylum, y = perf_metric_diff, fill = col)) + 
  geom_col(position = position_dodge(), color = "black") +
  scale_fill_identity() +
  annotation_custom(phylum.plot.legend,
                    ymax = 0.015,
                    ymin = 0.01,
                    xmax = 4,
                    xmin = 5) +
  theme(legend.position = "none",
        plot.margin = margin(10,10,10,10),
        axis.title.x = element_blank(),
        axis.text.x = element_text(color = "black"),
        axis.title.y = element_text(margin = margin(0,15,0,5), size = 13)) + 
  labs(x = "Phylum",
       y = "Δ Normalized RMSE") -> phylum.plot
  
#generate feature importance plot with the most 'important' genera for each outcome 
rbind(df.rich.fi,
      df.pl.fi) %>% 
  group_by(outcome) %>% 
  arrange(desc(perf_metric_diff)) %>% 
  slice_head(n = 10) %>% 
  ungroup() %>%
  left_join(data.frame(
    expand.grid(
      phylum = c("Proteobacteria", "Bacteroidetes", "Actinobacteria", "Firmicutes"),
      outcome = c("richness", "log_copy_number")),
    col = c("#ff8a9f", "#fdff8a", "#8aceff", "#8affa3", 
            "#a32d43", "#a1a32d", '#2d72a3', "#2da346")
  )) %>% 
  mutate(col = ifelse(is.na(col) & outcome == "richness", "gray90",
                      ifelse(is.na(col) & outcome == "log_copy_number", 'gray50', col)),
         genus = str_replace(case_when(grepl('unclassified', .$genus, ignore.case =T) ~ paste0("Unclassified<span style = 'color:#ffffff;'>.</span>", .$genus),
                                    !grepl('unclassified', .$genus, ignore.case =T) ~ paste0('<i>', .$genus, '</i>')), 
                             "_unclassified",
                             ""),
         genus = str_replace_all(genus, "_", " "), 
         genus = fct_reorder2(genus, outcome, perf_metric_diff),
         outcome = str_to_title(fct_recode(outcome, `Pathogen Load` = 'log_copy_number'))) %>% 
  ggplot(aes(x = genus, y = perf_metric_diff)) +
  geom_col(aes(fill = col), color = "black") +
  scale_fill_identity('col') +
  scale_y_continuous(expand = c(0,0), limits = c(0,0.0034)) +
  facet_wrap(~outcome, scales = "free_x") + 
  theme(axis.text.x = element_markdown(size = 10,
                                   angle = 45, vjust = 1, hjust=1,
                                   color = "black"),
        plot.margin = margin(10,10,10,10),
        axis.title.x = element_blank(),
        axis.title.y = element_text(margin = margin(0,15,0,5), size = 13),
        strip.text = element_text(size = 15)) + 
  labs(x = "Genus",
       y = "Δ Normalized RMSE") -> genus.plot

#clean up working environment
keep.objs <- c(
  keep.objs,
  "genus.plot",
  "phylum.plot",
  'pl.fi',
  "rich.fi")
remove.objs <- subset(ls(), !(ls() %in% keep.objs)) %>%
  subset(. != "keep.objs")
rm(list = remove.objs)

# ANCOMBC2 and Dysbiosis Index #####

#
#begin by generating a phyloseq object
#

# format sample metadata
sample.data <- beta.meta %>% 
  column_to_rownames(var = "swab_label") %>%
  sample_data()

# format OTU abundance table
otu.table <- otu.mat %>%
  filter(rownames(.) %in% sample_names(sample.data)) %>% 
  as.matrix() %>%
  otu_table(taxa_are_rows = F) 

# format taxonomy table 
tax.table <- tax %>%
  filter(otu %in% taxa_names(otu.table)) %>%
  column_to_rownames(var="otu") %>%
  as.matrix() %>%
  tax_table()

#merge formatted dataframes into phyloseq object
spatial.phylo <- merge_phyloseq(otu.table, sample.data, tax.table)
gc() #clear memory extraneous usage 

# Run ANCOM model which examines effect of pathogen presence on taxa abundance  
# ancom.model.qpcr <- ancombc2(
#  data = spatial.phylo,  #input data via phyloseq object
#  fix_formula = "oo_present", #qPCR (+/-) as fixed effect
#  rand_formula = "(1|county)", #county used as random intercept
#  tax_level = 'genus', #microbial genus used as response level
#  group = "ecomode",  #samples grouped by host ecomode to detect structural zeros
#  prv_cut = 0.05, # Taxa with prevalence less than prv_cut will be excluded in the analysis. Default is 10 %
#  struc_zero = T, #Taxa which are not observed in a group are treated as not occurring in that group
#  n_cl = 31, #Set number of cores to be used in parallel processing
#  verbose = T, #report each step in computation to console
#  iter_control = list(tol = 0.01, max_iter = 20, verbose = T) #report each REML iteration to console
# )
# saveRDS(ancom.model.qpcr, file = "ancom.model.qpcr.rds")
ancom.model.qpcr <- readRDS(file = "ancom.model.qpcr.rds")

# Run ANCOM model which examines effect of pathogen load on taxa abundance  
# ancom.model.pl <- ancombc2(
#   data = spatial.phylo,  #input data via phyloseq object
#   fix_formula = "log_copy_number", #qPCR (+/-) as fixed effect
#   rand_formula = "(1|county)", #county used as random intercept
#   tax_level = 'genus', #microbial genus used as response level
#   group = "ecomode",  #samples grouped by host ecomode to detect structural zeros
#   prv_cut = 0.05, # Taxa with prevalence less than prv_cut will be excluded in the analysis. Default is 10 %
#   struc_zero = T, #Taxa which are not observed in a group are treated as not occurring in that group
#   n_cl = 31, #Set number of cores to be used in parallel processing
#   verbose = T, #report each step in computation to console
#   iter_control = list(tol = 0.01, max_iter = 20, verbose = T) #report each REML iteration to console
# )
# saveRDS(ancom.model.pl, file = "ancom.model.pl.rds")
ancom.model.pl <- readRDS(file = "ancom.model.pl.rds")
  
#generate list of genera (and their OTUs) who have differential abundance 
#based on pathogen presence / abundance 
rbind(
  ancom.model.qpcr$res %>% 
  select(-contains("Intercept")) %>% 
  filter(if_any(contains('diff_'))) %>% 
  mutate(diff = ifelse(.[,2] > 0, "positive", "negative")) %>% 
  select(taxon, diff)
,
ancom.model.pl$res %>%
  select(-contains("Intercept")) %>%
  filter(if_any(contains('diff_'))) %>%
  mutate(diff = ifelse(.[,2] > 0, "positive", "negative")) %>%
  select(taxon, diff)
) %>%
  rename(genus = taxon) %>% 
  left_join(tax, by = 'genus', relationship = "many-to-many") %>% 
  select(diff, genus, otu) %>% 
  arrange(diff, genus, otu) -> dys.taxa

# calculate Dysbiosis index according to Gevers et al. (2014)
# doi:10.1016/j.chom.2014.02.005
df %>% 
  select(swab_label, oo_present, log_copy_number, any_of(dys.taxa$otu)) %>% 
  pivot_longer(cols = any_of(dys.taxa$otu), names_to = "taxa", values_to = "reads") %>% 
  left_join(., dys.taxa, by = c("taxa" = "otu")) %>%
  select(-genus) %>% 
  group_by(swab_label, diff) %>% 
  mutate(reads = sum(reads)) %>% 
  select(-taxa) %>% 
  sample_n(1) %>% 
  ungroup() %>% 
  pivot_wider(names_from = diff, values_from = reads) %>% 
  #slight modification here (pseudocount to accommodate log transformation (+1))
  mutate(
    positive = positive + 1, 
    negative = negative + 1) %>% 
  #dysbiosis index is calculated here 
  mutate(dysbiosis.index = log(positive/negative)) %>% 
  select(swab_label, oo_present, log_copy_number, dysbiosis.index) %>% 
  as.data.frame() -> df.dysbiosis.index

#use Youden's J statistic to select diagnostic cutoff value  
optimal.cutpoints(X = "dysbiosis.index",
                  status = "oo_present",
                  tag.healthy = "0",
                  methods = "Youden",
                  data = df.dysbiosis.index) -> y.index
str(summary(y.index))

#generate figure to illustrate diagnostic value of dysbiosis index 
df.dysbiosis.index %>% 
  mutate(oo_present = fct_recode(oo_present,
                                 `Positive` = '1',
                                 `Negative` = '0')) %>% 
  ggplot(aes(x = oo_present, y = dysbiosis.index)) + 
  geom_violin(aes(fill = oo_present), color = 'black', linewidth = 0.75, width = 1) + 
  geom_boxplot(width = 0.25, fill = NA, color = 'white', linewidth = 0.75) +
  geom_hline(aes(yintercept = summary(y.index)$Youden$Global$optimal.cutoff$cutoff),
             linewidth = 1.5,
             alpha = 0.85,
             color = 'black',
             lty = "longdash") + 
  annotate(geom = "text",
           y = summary(y.index)$Youden$Global$optimal.cutoff$cutoff+0.75,
           x = 0, 
           size = 3,
           label = paste("Cutoff =", signif(summary(y.index)$Youden$Global$optimal.cutoff$cutoff, 2)),
           hjust = 0) +
  annotate(geom = "text",
           y = 8,
           x = 0, 
           size = 5,
           label = paste0("NPV = ", signif(summary(y.index)$Youden$Global$optimal.cutoff$NPV, 3),
                          "\n",
                          "PPV = ", signif(summary(y.index)$Youden$Global$optimal.cutoff$PPV, 3)),
           hjust = 0) +
  scale_x_discrete(expand=c(0,1.1)) +
  scale_fill_brewer(palette = "Set1", direction = -1) + 
  theme(legend.position = "none",
        axis.text.x = element_text(size = 15, color = "black"),
        axis.text.y = element_text(size = 15, color = "black"),
        plot.margin = margin(10,10,10,10),
        axis.title.y = element_text(margin = margin(0,15,0,5)),
        axis.title.x = element_text(margin = margin(15,0,5,0))) + 
  labs(x = "qPCR Detection",
       y = "Dysbiosis Index") -> dys.fig

#generate multipanel of all taxa specific analyses 
ggarrange(phylum.plot,
          ggarrange(
            genus.plot,
            dys.fig,
            ncol = 2,
            widths = c(1.3, 1),
            labels = c("B", "C"),
            font.label = list(size = 20)
          ),
          nrow = 2,
          heights = c(1, 1.5),
          labels = c("A", ""),
          font.label = list(size = 20)) -> tax.multi

#save taxa specific analyses multipanel 
ggsave(plot = tax.multi,
       filename = 'tax.multi.jpg',
       device = 'jpg', 
       units = 'px',
       height = 2500,
       width = 4000,
       path = str_replace(getwd(), "/datafiles", "/Figures"))
