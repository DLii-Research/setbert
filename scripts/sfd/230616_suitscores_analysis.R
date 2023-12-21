#load required packages
library(tidyverse)
library(lme4)
library(lmerTest)
library(glmmTMB)
library(MuMIn)
library(ggtext)
library(ggpubr)
library(rcompanion)
library(Boruta)
library(janitor)
library(jpeg)
library(grid)
library(ggpubr)
library(patchwork)

#set working directory
setwd('G:/My Drive/Walker lab/SFD Spatial Dataset/Manuscript code.2/datafiles')

#set plotting theme 
theme_set(theme_classic() + 
            theme(panel.grid.major = element_blank(), 
                  panel.grid.minor = element_blank(),
                  panel.border = element_blank(),
                  axis.line = element_line(colour = "black"),
                  axis.text.y   = element_text(size=12),
                  axis.text.x   = element_text(size=12),
                  axis.title.y  = element_text(size=14, margin = margin(r = 10, l = 10)),
                  axis.title.x  = element_text(size=14, margin = margin(t = 10, b = 10)),
                  legend.text = element_text(size=11),
                  legend.title=element_text(size=13),
                  plot.title = element_text(size=21, hjust = 0.5),
                  legend.position = "top", 
                  legend.background = element_rect(fill="white", colour = NA),
                  plot.margin = margin(t = 10, r = 10))) 

#load suitability score data 
df.suit <- read_rds('df.suit.rds') %>% 
  #rename columns for ease of use
  rename('dys' = 'dysbiosis.index',
         'suit' = 'OoSuitability') %>% 
  #transform suitability scores to P/A using various thresholds  
  mutate(
    suit.max = as.factor(ifelse(suit > 0.668684, 1 , 0)), #Maximum sensitivity plus specificity 
    suit.equal = as.factor(ifelse(suit > 0.59222, 1 , 0)), #Equal Sensitivity plus specificity 
    suit.sen = as.factor(ifelse(suit > 0.297813, 1 , 0)) #Sensitivity
  ) %>% 
  mutate(across(.cols = contains('suit.'), ~fct_inseq(.x)))

filter(df.suit, site %in% hss) %>%
  group_by(site) %>% 
  sample_n(2) %>% 
  arrange(site) 

df.suit %>% 
  group_by(site) %>% 
  filter(n() > 10) %>% 
  group_by(site, genus) %>% 
  filter(n() > 3) %>% 
  group_by(site) %>% 
  summarise(host_richness = n_distinct(genus),
            n_samples = n()) %>% 
  arrange(host_richness, decreasing = F)

df.suit %>% 
  group_by(site) %>% 
  filter(n() > 10) %>% 
  ungroup() %>% 
  mutate(site = fct_reorder(site, suit, .desc = T)) %>% 
  ggplot(aes(x = site, y = suit)) +
  geom_boxplot()

df.suit %>% 
  filter(site %in% sites.compare) %>% 
  group_by(site, genus) %>% 
  filter(n() > 3) %>% 
  ungroup() -> plot.df

plot.df %>% 
  group_by(site, genus) %>% 
  summarize(dys = mean(dys)) %>% 
  arrange(site, desc(dys)) %>% 
  pull(genus) %>%
  unique() %>% 
  as.character() -> x.axis.levels
  
plot.df %>%
  mutate(genus = factor(genus, levels = x.axis.levels)) %>%
  group_by(site, genus) %>% 
  summarize(dys.mean = mean(dys),
            dys.se = sd(dys)/sqrt(n())) %>% 
  ggplot(aes(x = genus, y = dys.mean)) + 
  geom_errorbar(aes(ymin = dys.mean - dys.se, ymax = dys.mean + dys.se), width = 0.25) +
  geom_col(aes(fill = site), color = 'black', show.legend = F) + 
  facet_wrap(~site, scales = 'free_x')
  
  summarize(n = n()) %>% 
  arrange(site, desc(n))

#
# Perform GLMM modeling with Threshold Suitability as predictor 
#

#start with dysbiosis index as the response variable is much easier to model 

#dysbiosis index is normally distributed so a generalized model is not required 
plotNormalDensity(df.suit$dys)
shapiro.test(df.suit$dys)

#use boruta to determine what variables might need to be included as random effects 
#run boruta
df.suit %>%
  select(-c(swab_label, suit, date, gpsn, gpsw, day,
            dayofyear, year_f, time, temperature_c, species,
            sex, gravid, lesion_type, lesion_location, lesion_number,
            lesion_coverage, photos, blood_smear, blood_sample, ct, copies_rxn,
            swab_plate_n, notes, tn_ecoregion)) %>% 
  drop_na() %>% 
  Boruta(dys ~ ., data = ., doTrace = 2) -> boruta.dys
#examine boruta results 
plot(boruta.dys)
boruta.dys$ImpHistory %>% 
  t() %>% 
  as.data.frame() %>% 
  rownames_to_column("feature") %>% 
  filter_at(vars(contains("V")), ~.x > -Inf) %>% 
  mutate(Means = rowMeans(.[,-1])) %>% 
  select(-contains("V")) %>% 
  arrange(desc(Means)) %>% 
  head(n = 15)
#looks like ecomode, genus, site, disease severity, and oo_present are 
#all good canidates for random effects 

#perform AIC model selection with varied random effects structure
m1.1 <- lmer(dys ~ suit.max + (1|site), data = df.suit) 
m1.2 <- lmer(dys ~ suit.max + (1|site) + (1|genus), data = df.suit) 
m1.3 <- lmer(dys ~ suit.max + (1|site) + (1|genus) + (1|ecomode), data = df.suit) 
m1.4 <- lmer(dys ~ suit.max + (1|site) + (1|genus) + (1|ecomode) + (1|dis.sev), data = df.suit) 
m1.5 <- lmer(dys ~ suit.max + (1|site) + (1|genus) + (1|ecomode) + (1|oo_present), data = df.suit) 
AIC(m1.1, m1.2, m1.3, m1.4, m1.5)

#examine fit of best model
plot(m1.4)
qqnorm(resid(m1.4))
qqline(resid(m1.4))
#looks good! 

#examine results of best fitting model
summary(m1.4)
r.squaredGLMM(m1.4)
#dysbiosis index (in its current state) is not predicted by the suitability threshold 

#Response of dysbiosis index to suitability threshold 
ggplot(df.suit, aes(x = suit.max, y = dys)) + 
  geom_violin(aes(fill = suit.max), adjust = 1.5, show.legend = F, linewidth = 0.75) + 
  geom_boxplot(width = 0.05, alpha = 0.5, linewidth = 0.75, outlier.alpha = 1) +
  scale_x_discrete(limits = c('0', '1'),
                   labels = c("Not Suitable",
                              "Suitable")) + 
  scale_fill_manual(values = c('#a4db76', '#f88588')) +
  geom_richtext(aes(x = 1.15,
                    y = 7.5,
                    label = "p = 0.74<br>R<sup>2</sup> = 1.6x10<sup>-4</sup>"),
                hjust = 0,
                size = 3,
                label.colour = NA,
                fill = NA) + 
  labs(y = "Dysbiosis index",
       x = "Threshold Suitability") + 
  theme(axis.title.x = element_markdown(margin = margin(t = 10, b = 10)),
        axis.title.y = element_markdown(margin = margin(r = 10, l = 10)),
        axis.text = element_text(color = "black"),
        plot.margin = margin(30,30,10,10)) -> fig.suit_di
 # ggsave(filename = "dys_suit.jpg", device = "jpg", units = "px", width = 3000, height = 2000, path = "G:/My Drive/Walker Lab/SFD Spatial Dataset/Manuscript code.2/Figures")

#proceed with pathogen load modeling 

#Pathogen load has a very problematic distribution to model 
table(df.suit$oo_present) #massive proportion of zeros 
plotNormalDensity(df.suit$log_copy_number) #hugely right skewed distribution  

#use boruta to determine what variables might need to be included as random effects 
#run boruta
df.suit %>%
  select(-c(swab_label, suit, date, gpsn, gpsw, day,
            dayofyear, year_f, time, temperature_c, species,
            sex, gravid, lesion_type, lesion_location, lesion_number,
            lesion_coverage, photos, blood_smear, blood_sample, ct, copies_rxn,
            dis.sev, oo_present, swab_plate_n, notes, tn_ecoregion)) %>% 
  drop_na() %>% 
  Boruta(log_copy_number ~ ., data = ., doTrace = 2) -> boruta.lcn
#examine boruta results 
plot(boruta.lcn)
boruta.dys$ImpHistory %>% 
  t() %>% 
  as.data.frame() %>% 
  rownames_to_column("feature") %>% 
  filter_at(vars(contains("V")), ~.x > -Inf) %>% 
  mutate(Means = rowMeans(.[,-1])) %>% 
  select(-contains("V")) %>% 
  arrange(desc(Means)) %>% 
  head(n = 15)
#looks like ecomode, genus, site are all good canidates for random effects 

#perform AIC model selection with varied random effects structure
m2.1 <- glmmTMB(copies_rxn ~ suit.max + (1|ecomode), data = df.suit, ziformula = ~1, family = poisson)
m2.2 <- glmmTMB(copies_rxn ~ suit.max + (1|ecomode) + (1|genus), data = df.suit, ziformula = ~1, family = poisson)
m2.3 <- glmmTMB(copies_rxn ~ suit.max + (1|ecomode/genus), data = df.suit, ziformula = ~1, family = poisson)
m2.4 <- glmmTMB(copies_rxn ~ suit.max + (1|site), data = df.suit, ziformula = ~1, family = poisson, control = glmmTMBControl(optimizer = optim, optArgs = list(method="BFGS")))
AIC(m2.1, m2.2, m2.3, m2.4)

#examine fit of best model
qqnorm(resid(m2.4))
qqline(resid(m2.4))
#model fit could be better but given the underlying distribution
#I think this is as good as we're gonna get 

#examine results of best fitting model
summary(m2.4)
r.squaredGLMM(m2.4)
#pathogen load is predicted by the suitability threshold 

#Response of pathogen load to suitability threshold 
ggplot(df.suit, aes(x = suit.max, y = copies_rxn+1)) + 
  geom_violin(aes(fill = suit.max), adjust = 1.5, show.legend = F, linewidth = 0.75) + 
  geom_boxplot(width = 0.05, alpha = 0.5, linewidth = 0.75, outlier.alpha = 1) +
  scale_x_discrete(labels = c("Not Suitable",
                              "Suitable")) + 
  scale_y_log10(labels = c('0', '10', '100', '1000', '10000'), breaks = c(1, 11, 101, 1001, 10001)) +
  scale_fill_manual(values = c('#a4db76', '#f88588')) +
  labs(y = "Pathogen Load",
       x = "Threshold Suitability") + 
  geom_richtext(aes(x = 1.1,
                    y = 10001,
                    label = "p < 2x10<sup>-16</sup><br>R<sup>2</sup> = 0.042",
                    hjust = 0),
                size = 3,
                label.colour = NA,
                fill = NA
                ) + 
  theme(axis.title.x = element_markdown(margin = margin(t = 10, b = 10)),
        axis.title.y = element_markdown(margin = margin(r = 10, l = 10)),
        axis.text = element_text(color = "black"),
        plot.margin = margin(30,30,10,10)) -> fig.suit_pl
# ggsave(filename = "dys_pathogenload.jpg", device = "jpg", units = "px", width = 3000, height = 2000, path = "G:/My Drive/Walker Lab/SFD Spatial Dataset/Manuscript code.2/Figures")

thresh.map <- rasterGrob(readJPEG("C:\\Users\\aromer\\Downloads\\230609_Thresh_Map.jpg")) 
suit.map <- rasterGrob(readJPEG("C:\\Users\\aromer\\Downloads\\230609_Cont_Map.jpg")) 

ggarrange(fig.suit_pl,
          fig.suit_di,
          labels = c("A", "B"),
          font.label = list(size = 20),
          label.x = -0.08) /
  thresh.map + suit.map + 
  plot_layout() +
  plot_annotation(tag_levels = list(c("", "C", "D"))) &
  theme(plot.tag = element_text(size = 20,
                                face = "bold",
                                color = "black",
                                family = NULL),
        plot.margin = margin(10,10,10,10)) -> spatial.multi

ggsave(spatial.multi,
       filename = "test.jpg",
       path = "C:/Users/aromer/Downloads",
       height = 3000,
       width = 2500,
       dpi = 300,
       units = "px")

