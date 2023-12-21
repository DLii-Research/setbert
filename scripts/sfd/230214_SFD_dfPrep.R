# Prepare coding environment & load data #####

#load required packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(ggpubr)
  library(data.table)
  library(decontam)
  library(vegan)
  library(ggtext)
  library(PERFect)
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

#set working directory and random seed
set.seed(11301996)
setwd("C:/Users/aromer/My Drive/Walker Lab/SFD Spatial Dataset/Manuscript code.2/datafiles")

#get date for file saving 
date <- format(Sys.Date(), "%y%m%d")

#read in raw data

#save file names as objects 
otu.file <- "221205_cmfp.trim.contigs.pcr.good.unique.good.filter.unique.precluster.denovo.vsearch.pick.opti_mcc.shared"
tax.file <- "221205_cmfp.trim.contigs.pcr.good.unique.good.filter.unique.precluster.denovo.vsearch.pick.opti_mcc.0.03.cons.taxonomy"
seq.meta <- "221216_seqmeta.full_ASR.csv"

#load and format non-rarefied and non-decontaminated OTU data file
otu.mat <- fread(otu.file, showProgress = T) %>% #read shared file from mothur 
  select(-c(label, numOtus)) %>% #remove bioinformatics metadata 
  as.data.frame() %>%  #store as dataframe 
  column_to_rownames("Group") #ensure dataframe contains only numeric columns 

#load and format Otu taxonomy file 
tax <- fread(tax.file) %>% #read taxonomy file 
  select(-Size) %>% #remove n reads per otu
  rename(otu = OTU) %>% #rename OTU to make coding easier
  #remove bootstrap values provided by mothur
  mutate(Taxonomy = gsub("\\s*\\([^\\)]+\\)","", Taxonomy)) %>%  
  #create seperate columns for each taxonomic classification
  separate(Taxonomy,
           sep = ";",
           into = c("kingdom", "phylum", "class", "order", "family", "genus"),
           extra = "drop") %>% 
  #retain only otus with sequencing data (in taxonomy file) 
  filter(otu %in% names(otu.mat)) 

#sequencing metadata file
df.seqmeta <- fread(seq.meta, stringsAsFactors = T)

#clean up working environment
keep.objs <- c("otu.mat",
               "tax",
               "df.meta",
               "df.seqmeta",
               "select",
               "filter",
               "date")
remove.objs <- subset(ls(), !(ls() %in% keep.objs)) %>% 
  subset(. != "keep.objs")
rm(list = remove.objs) 
  
# Remove rare taxa #####

#filter rare taxa using abundance cutoff 
otu.mat %>% 
  select(where( ~ is.numeric(.x) && sum(.x) > 10)) %>% 
  as.matrix() -> otu.mat.abund

#Remove rare taxa using PERFect at alpha = 0.001
#PERFout <- PERFect_sim(X = otu.mat.abund, alpha = 0.001)
#saveRDS(PERFout, file = paste(date, "SFD_PERFout_a=.001.rds", sep = "_"))
PERFout <- readRDS("230213_SFD_PERFout_a=.001.rds")

#extract OTU matrix from PERFect object 
otu.mat.perf <- PERFout$filtX %>% 
  #convert to dataframe
  as.data.frame() %>% 
  #arrange columns by OTU#
  .[,order(colnames(.))]

#clean up working environment
keep.objs <- subset(keep.objs, !keep.objs %in% c("otu.mat")) %>% 
  c(., "otu.mat.perf")
remove.objs <- subset(ls(), !(ls() %in% keep.objs)) %>% 
  subset(. != "keep.objs")
rm(list = remove.objs) 

# Perform decontamination ####
  
#Variable that determines if computationally expensive calculation should be repeated
calculate <- "Y"
#file naming parameters
environ.desig <- "decontam.out" #file name in R environment
file.suffix <- "decontam.out.rds" #file suffix in windows environment 

#either load previous iteration or perform new computation based on user input 
if(calculate == "N"){
  #first check if they file already exists in R environment
  if(environ.desig %in% ls()){
    print("file already exists in Environment")
    #next check if there is a file created today in the working directory
  } else if (paste(date, file.suffix, sep = "_") %in% list.files()){
    assign(environ.desig, readRDS(paste(date, file.suffix, sep = "_")))
    print("loaded file (generated today) from working directory")
    #next load the last file that was generated
  } else if (any(grepl(file.suffix, list.files()))) {
    #get the date the last file was generated
    subset(list.files(), grepl(file.suffix, list.files())) %>%
      str_replace(., file.suffix, "") %>%
      str_replace(., "_", "") %>%
      subset(., . != "") %>%
      as.Date(., format = "%y%m%d") %>%
      sort() %>%
      tail(n = 1) %>%
      format("%y%m%d") -> lastfile.date
    #use it to get the file name
    subset(list.files(), grepl(lastfile.date, list.files())) %>%
      subset(., grepl(file.suffix, .)) -> lastfile.name
    #load file from working directory
    assign(environ.desig, readRDS(lastfile.name))
    print(paste("loaded file (generated ",as.Date(lastfile.date, format = "%y%m%d"),") from working directory", sep = ""))
    #finally indicate there is no file to be loaded if that is the case 
  } else {
    print("saved objection representation does not exist in working directory")    
  }
} else if(calculate == "Y"){
  #conduct decontamination via decontam
  print("Performing decontamination")
  
  #Reorder sequence metadata to match OTU matrix 
  otu.mat.perf %>%
    rownames_to_column("swab_label") %>%
    select(c(1)) %>% 
    left_join(.,
              df.seqmeta) -> df.seqmeta 
  
  #join OTU feature table with sequencing metadata to perform decontamination
  otu.mat.perf %>% 
    rownames_to_column("swab_label") %>% 
    right_join(df.seqmeta, ., by = "swab_label") %>% 
    #remove any samples without a known DNA concentration
    filter(!is.na(DNA_con)) %>%
    #annotate samples are NTC or actual samples with boolean
    mutate(sample_type = (grepl("neg", swab_label, ignore.case = T)|
                            grepl("nc", swab_label, ignore.case = T)),
           .after = swab_label) -> otus.seqdat
  
  #remove rownames - artefact from OTU matrix
  row.names(otus.seqdat) <- NULL
  
  #specify sequence of threshold probability values to iterate through 
  threshold <- c(0.05, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999)

  #generate empty lists for loop output items 
  sapply(c("decontam.results.list",
           'decontam.summary',
           "plots.sampletype",
           "plots.persample"),
         function(k) assign(k, list(), envir = .GlobalEnv))
  
  #format #sequence data for decontam analysis
  otus.seqdat %>% 
    column_to_rownames("swab_label") %>% 
    select(starts_with("Otu")) %>% 
    as.matrix() -> decontam.seqs 
  
  # conduct decontamination across range of threshold values 
  for(j in 1:length(threshold)){
    
    #report threshold value to track loop 
    print(paste("Threshold:", threshold[j], sep = " "))
    
    #create var for sys.time() and track time
    t.start <- Sys.time()
    
    #identify contaminating taxa via decontam 
    decontam.results <- isContaminant(
      seqtab = decontam.seqs,
      neg = otus.seqdat$sample_type,
      conc = otus.seqdat$DNA_con,
      method = "combined",
      threshold = threshold[j],
      normalize = TRUE,
      detailed = T) %>%
      #retain only contaminating taxa 
      filter(contaminant == T) %>% 
      #add taxonomy data to results 
      rownames_to_column("otu") %>%
      left_join(., tax, by = "otu")
    
    #save results into list
    decontam.results.list[[j]] <- decontam.results
    
    #name each dataframe according to the data set and threshold value 
    names(decontam.results.list)[j] <- paste("Threshold", threshold[j], sep=": ")
    
    #extract list of contaminant OTUs 
    contaminants <- decontam.results.list[[j]]$otu
    
    #generate long dataframe and classify taxa according to decontam results 
    df.otu.long <- otus.seqdat %>%
      pivot_longer(starts_with("Otu"), names_to = "otu", values_to = "reads") %>% 
      mutate(otu.type = ifelse(otu %in% contaminants, "contamination", "sample taxa")) %>% 
      group_by(sample_type) %>% 
      mutate(reads = as.numeric(reads),
             total.reads = sum(reads))  %>% 
      ungroup() %>% 
      group_by(swab_label) %>% 
      mutate(sample.reads = sum(reads)) %>%
      ungroup() %>% 
      mutate(otu.type = as.factor(str_to_title(otu.type)))
    
    #Ensure plots always have a full legend 
    if(!("contamination" %in% levels(df.otu.long$otu.type))){
      levels(df.otu.long$otu.type)[length(levels(df.otu.long$otu.type))+1] <- "Contamination"
    }
    
    #plot decontam results according to sample type 
    #generate dataframe for plotting
    df.plot.sampletype <- df.otu.long %>%  
      group_by(swab_label, otu.type) %>%
      mutate(otu.id.reads = sum(reads),
             otu.id.proportion = otu.id.reads/sample.reads) %>%
      sample_n(1) %>%   
      group_by(sample_type, otu.type) %>% 
      mutate(mean.otu.id.pro = mean(otu.id.proportion),
             se.otu.id.pro = sd(otu.id.proportion)/sqrt(length(otu.id.proportion))) %>%
      mutate(sample_type = ifelse(sample_type, "No Template Control", "Sample")) %>% 
      ungroup()
    #render plot
    plot.sampletype <- df.plot.sampletype %>% 
      ggplot(aes(x = sample_type, y = -mean.otu.id.pro, fill = otu.type)) +
      geom_errorbar(aes(ymax = -(mean.otu.id.pro + se.otu.id.pro),
                        ymin = -(mean.otu.id.pro - se.otu.id.pro)),
                    position = position_dodge(width = 0.9), width = 0.25) +
      geom_col(color = "black", position = position_dodge(width = 0.9)) + 
      scale_y_reverse(limits = c(0,-1), labels = scales::percent) +
      scale_fill_manual(drop = FALSE,
                        values = c("Contamination" = "#F8766D", "Sample Taxa" = "#00BFC4")) +
      labs(title = paste("Threshold =", threshold[j], sep = " "),
           x = "Sample Type",
           y = "Sample Composition") + 
      theme_classic() +
      theme(plot.title = element_text(size = 12),
            legend.title = element_blank())
    
    #extract summary values for decontamination
    decontam.summary[[j]] <- df.plot.sampletype %>%
      mutate(threshold = threshold[j]) %>%
      as.data.frame()
    
    #save plots in a named list
    plots.sampletype[[j]] <- plot.sampletype
    names(plots.sampletype)[j] <- paste("Threshold", threshold[j], sep=": ")
    
    #plot decontam results on a per-sample basis 
    df.persample <- df.otu.long %>%
      group_by(swab_label) %>% 
      mutate(sample.reads = sum(reads)) %>% 
      ungroup() %>%
      mutate(swab_label = as.factor(swab_label),
             sample.reads = as.numeric(sample.reads),
             swab_label = fct_reorder(swab_label, sample.reads, .desc = T)) %>%
      group_by(swab_label, otu.type) %>%
      mutate(read.counts = sum(reads)) %>% 
      sample_n(1) %>% 
      select(-otu) 
    
    plot.persample <- df.persample %>%
      filter(read.counts > 0) %>% 
      ggplot(aes(x = swab_label, y = read.counts, fill = otu.type)) + 
      geom_col(aes(y = read.counts-100000, x = swab_label, fill = otu.type), color = "black", alpha = 1, linewidth = 0.5) + 
      geom_col(width = 1, show.legend = F) +
      scale_y_continuous(expand = c(0,0)) + 
      coord_cartesian(ylim = c(0,as.numeric(quantile(df.persample$read.counts, 0.999)))) + 
      labs(title = paste("Threshold =", threshold[j], sep = " ")) + 
      scale_fill_manual(values = c("Contamination" = "#F8766D", "Sample Taxa" = "#00BFC4")) +
      theme_classic() +
      theme(axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            plot.title = element_text(size = 12))
    
    #save plots in a named list
    plots.persample[[j]] <- plot.persample
    names(plots.persample)[j] <- paste("Threshold", threshold[j], sep=": ")
    
    #report time to track loop 
    t.deco <- abs(t.start-Sys.time())
    print(paste("Loop", " ", j, "/", length(threshold), "    ", "Elapsed time: ", round(t.deco, 4), sep = ""))
    
  }

  # generate plot to select optimal threshold value by examining
  # how many taxa are removed at each step change in threshold value
  summary.df <- decontam.summary %>% 
    bind_rows() %>%
    select(-c(otu, reads)) %>% 
    filter(otu.type == "Contamination") %>% 
    group_by(threshold, sample_type) %>% 
    mutate(sample.type.reads = total.reads, 
              thresh.contam.reads = sum(otu.id.reads)) %>% 
    sample_n(1) %>%
    ungroup() %>% 
    mutate(thresh.contam.pro = thresh.contam.reads/sample.type.reads,
           delta.thresh.contam = -(thresh.contam.pro - shift(thresh.contam.pro, n = 2))) %>% 
    mutate(delta.thresh.contam = ifelse(is.na(delta.thresh.contam), thresh.contam.pro, delta.thresh.contam))
  summary.df %>% 
    ggplot(aes(x = as.factor(threshold), y = delta.thresh.contam, fill = sample_type)) +
    geom_col(position = position_dodge(), color = "black") +
    scale_y_reverse(labels = scales::percent) +
    labs(y = "&Delta; Total Reads",
         x = "Threshold Value") +
    scale_fill_brewer(palette = "Paired") +
    theme_classic() +
    theme(legend.title = element_blank(), 
          legend.position = "bottom", 
          text = element_text(size = rel(4.5)),
          legend.text = element_text(size = 14),
          legend.spacing.x = unit(0.5, 'cm'),
          axis.title.x = element_text(margin = margin(10,0,0,0)),
          axis.title.y = ggtext::element_markdown(margin = margin(0,15,0,10))
          ) -> decontam.summary.plot
  
  #store decontam output in tidy list
  decontam.out <- list(
    'decontam.results.list' = decontam.results.list,
    'plots.persample' = plots.persample, 
    'plots.sampletype' = plots.sampletype,
    'decontam.summary' = decontam.summary,
    'summary.df' = summary.df,
    'decontam.summary.plot' = decontam.summary.plot
  )
  
  #save decontam output as an .rds object to prevent recursive computational costs
  saveRDS(decontam.out, file = paste(date, "decontam.out.rds", sep = "_"))
  
}
  
#examine results of decontamination
(decontam.summary.plot<-decontam.out$decontam.summary.plot)

#analytically select threshold for final decontamination 
#
# i.e., first threshold where proportionally more seqs are removed from NTC than sample libraries 
# after removing 10% of ntc library seqs (background noise) 
decontam.out$summary.df %>% 
  group_by(sample_type) %>% 
  mutate(reads.remove.cumulative = cumsum(thresh.contam.pro)) %>% 
  select(threshold, sample_type, thresh.contam.pro, reads.remove.cumulative) %>% 
  mutate(ntc.reads.remove.cumulative = ifelse(sample_type == "No Template Control", reads.remove.cumulative, NA)) %>% 
  group_by(threshold) %>% 
  fill(ntc.reads.remove.cumulative) %>% 
  ungroup() %>% 
  filter(ntc.reads.remove.cumulative > .1) %>% 
  select(threshold, sample_type, thresh.contam.pro) %>% 
  mutate(sample_type = as.factor(sample_type),
         sample_type = fct_recode(sample_type, `ntc` = "No Template Control", `sample` = "Sample")) %>% 
  pivot_wider(names_from = "sample_type", values_from = 3) %>% 
  filter(ntc > sample) %>% 
  arrange(threshold) %>% 
  slice_head(n = 1) %>% 
  pull(threshold) -> threshold.decontam
threshold.decontam.index <- paste("Threshold:", threshold.decontam)
  
#extract decontam results for selected threshold value
decontam.final <- decontam.out[["decontam.results.list"]][[threshold.decontam.index]]
  
#remove contaminant taxa from OTU abundance matrix
contaminants <- decontam.final$otu
otu.mat.perf.decontam <- select(otu.mat.perf, -any_of(contaminants))

#generate final figures for supplementary materials 
decontam.out$plots.persample[[threshold.decontam.index]] +
  labs(x = "Samples<br><span style = 'font-size:11pt'>(arranged by sequencing depth)</span>",
       y = "16s Reads") +
  theme(legend.title = element_blank(),
        plot.title = element_blank(), 
        legend.position = c(0.8,0.9), 
        text = element_text(size = rel(4.5)),
        legend.text = element_text(size = 14),
        legend.spacing.x = unit(0.5, 'cm'),
        axis.title.x = element_markdown(margin = margin(10,0,0,0)),
        axis.title.y = element_text(margin = margin(0,15,0,10))
  ) -> persample.plot 

decontam.out$plots.sampletype[[threshold.decontam.index]] +
  labs(y = "% Reads<br><span style = 'font-size:12pt'>(Mean &plusmn; SE)</span>") +
  theme(legend.title = element_blank(),
        plot.title = element_blank(),
        legend.position = "none",
        text = element_text(size = rel(4.5)),
        legend.text = element_text(size = 14),
        legend.spacing.x = unit(0.5, 'cm'),
        axis.title.x = element_blank(),
        axis.title.y = element_markdown(margin = margin(0,15,0,10))
  ) -> sampletype.plot

decontam.summary.plot <- decontam.summary.plot +
  theme(plot.margin = margin(0,20,20,35),
        legend.position = c(0.1, 0.9))

#multipanel figure

ggpubr::ggarrange(
  decontam.summary.plot,
  ggpubr::ggarrange(sampletype.plot,
                    persample.plot,
                    ncol = 2,
                    align = "hv",
                    labels = c("b", 'c'),
                    font.label = list(size = 24, color = "black", face = "bold", family = "sans")
  ),
  labels = c("a", ''),
  font.label = list(size = 24, color = "black", face = "bold", family = "sans"),
  nrow = 2
) 
  
#clean up working environment
keep.objs <- subset(keep.objs, !keep.objs %in% c("otu.mat.perf", "decontam.out")) %>%
  c(., "otu.mat.perf.decontam") %>%
  unique()
remove.objs <- subset(ls(), !(ls() %in% keep.objs)) %>%
    subset(. != "keep.objs")
rm(list = remove.objs)
    
# Perform rarefaction & remove no template controls ####

#check the number of samples retained across a range of sequencing depths 
range.rarefy <- seq(0, 50)*1000
rarefy.summary <- list()
for(i in 1:length(range.rarefy)){
rarefy.depth <- range.rarefy[i]
otu.mat.perf.decontam %>% 
  as.data.frame() %>% 
  filter(rowSums(.) >= rarefy.depth) %>% 
  nrow() %>%
  as.numeric() -> samplesretained
data.frame(
  samples.retained = samplesretained,
  depth = rarefy.depth,
  per.samples.retained = samplesretained/nrow(otu.mat.perf.decontam)
) -> rarefy.summary[[i]]
}
do.call(rbind, rarefy.summary) -> rarefy.summary

#generate plot to select the most appropriate rarefaction depth 
rarefy.summary %>% 
  ggplot(aes(x = depth, y = per.samples.retained)) + 
  scale_y_continuous(labels = scales::percent, limits = c(0,1)) + 
  geom_segment(aes(x=min(depth),xend=max(depth),y=0.8647125,yend=0.8647125), lty = "dashed", color = "red", linewidth = 1) +
  geom_segment(aes(y=0,yend=1,x=10000,xend=10000), lty = "dashed" , color = "blue", linewidth = 1) + 
  geom_line(linewidth = 1.5) +
  labs(y = "Samples Retained",
       x = "Rarefaction Depth") + 
  theme_classic() +
  theme(text = element_text(size = rel(4.5)),
        axis.title.y = element_text(margin = margin(0,15,0,5)),
        axis.title.x = element_text(margin = margin(10,0,10,0)))

# 10,000 reads seems appropriate

#proceed with rarefaction
rare.depth <- 10000
suppressWarnings({otu.mat.perf.decontam %>% 
  rrarefy(rare.depth) %>%
  as.data.frame() %>% 
  filter(rare.depth == rowSums(.)) %>% 
  #remove any no template controls from the OTU table
  rownames_to_column("swab_label") %>% 
  mutate(sample_type = !(grepl("neg", swab_label, ignore.case = T)|
                          grepl("nc", swab_label, ignore.case = T))) %>% 
  filter(sample_type) %>% 
  select(-sample_type) %>% 
  mutate(swab_label = fct_recode(swab_label, `CMFP39.1` = "CMFP39")) %>% 
  column_to_rownames("swab_label") %>% 
  #store as matrix 
  as.matrix()}) -> otu.mat.rare

#collect swab labels for those samples which passed bioinformatics pipeline
final.sample.names <- rownames(otu.mat.rare)

#collect otu labels for those otus which passed bioinformatics pipeline
final.otus <- colnames(otu.mat.rare)

#subset taxonomy file a final time
tax.final <- filter(tax, otu %in% final.otus)

#clean up working environment
keep.objs <- keep.objs %>% 
  subset(!. %in% c('otu.mat.perf.decontam')) %>% 
  c("otu.mat.rare",
    "final.sample.names",
    "final.otus",
    "tax.final")
remove.objs <- subset(ls(), !(ls() %in% keep.objs)) %>%
  subset(. != "keep.objs")
rm(list = remove.objs)

# load and format metadata ####

#read in metadata file
metadata <- fread("201125_CMFP_master_georeferenced.csv", stringsAsFactors = T)

#all samples that passed bioinformatics are in master metadata files 
length(setdiff(rownames(otu.mat.rare), metadata$swab_label)) #Perfect! 

#subset metadata to just samples that passed bioinformatics 
metadata %>% 
  filter(swab_label %in% rownames(otu.mat.rare)) %>% 
  mutate(date = as.character(str_match(date, "(?<=/).*(?=/)")),
         swab_label = fct_recode(swab_label, `CMFP39` = "CMFP39.1")) %>% 
  left_join(df.seqmeta) %>% 
  #remove extraneous columns for ease of data handling 
  select(-c(swab_code, swab_number, original_code, scale_base, scale_apical,
            tail_length_cm, total_length_cm, gps_point, gps_n,
            gps_w, dna_extracted, microbiome_seq, walker_wash_swab_protocol)) %>%
  #remove DOA snakes due to small sample size to remove confounding variable 
  filter(doa == 0) %>% 
  select(-doa) -> doa.out
  
#Remove any recaptures to avoid repeat measures
  #remove recaps marked with PIT tags 
  doa.out %>% 
    filter(!is.na(pit_tag_id)) %>% 
    filter(grepl("3D", pit_tag_id)) %>% 
    group_by(pit_tag_id) %>% 
    arrange(year, month, date) %>% 
    slice_head(n = 1) -> pit.tag.unique 
  #remove recaps marked with cautery mark-recapture   
  doa.out %>% 
    filter(!is.na(pit_tag_id)) %>% 
    filter(!grepl("3D", pit_tag_id)) %>% 
    #while this animal was recaptured, the original sample is not in the dataset 
    mutate(recapture = ifelse(swab_label == "CMFP103", 0, recapture)) %>%
    filter(recapture == 0) -> cautery.unique 
  #rejoin data into common dataframe 
  doa.out %>% 
    filter(is.na(pit_tag_id)) %>%  
    rbind(pit.tag.unique,
          cautery.unique) %>% 
    select(-c(pit_tag_id, recapture)) -> no.recaps
#report n samples removed 
paste(nrow(doa.out) - nrow(no.recaps), "recapture samples removed") 

#load recovered metadata 
readxl::read_excel("missing_meta_sfd.xlsx") %>% 
  mutate(across(everything(), ~ifelse(.x %in% c("","NA"), NA, .x))) %>% 
  mutate(date = as.numeric(date),
         date = as.Date(date, origin = "1899-12-30")) %>% 
  mutate(across(.cols = c(missing, swab_plate, swab_well, qpcr), ~as.factor(.x))) %>% 
  mutate(across(.cols = c(gpsn, gpsw), ~as.numeric(.x))) %>% 
  select(-c(Notes)) -> rec.meta 

#restore as much metadata as was possible to recover 
no.recaps %>% 
  mutate(qpcr = as.factor(qpcr)) %>% 
  rename(day = date) %>% 
  left_join(., rec.meta, by = "swab_label") %>% 
  mutate(gpsn = coalesce(gpsn.x, gpsn.y),
         gpsw = coalesce(gpsw.x, gpsw.y),
         qpcr = coalesce(qpcr.x, qpcr.y)) %>% 
  select(-contains(c(".y", ".x"))) %>% 
  mutate(date = ifelse(!is.na(date), as.character(date),
                       ifelse(!is.na(day) & !is.na(month) & !is.na(year), paste(year, month, day, sep = "-"), 
                              ifelse(is.na(day) | is.na(month) | is.na(year), NA, "error"))),
         date = lubridate::as_date(date)) %>% 
  relocate(date, .after = year) %>% 
  filter(qpcr == 1) %>% 
  select(-c(missing, qpcr)) -> all.meta

#function that converts gps to county
source("C:/Users/aromer/My Drive/Walker Lab/custom_r_functions/latlong2county.R")

#get county locations using gps coordinations 
all.meta %>% 
  select(swab_label, gpsw, gpsn) %>% 
  filter(!is.na(gpsw)) %>% 
  mutate(county = unlist(str_extract_all(latlong2county(data.frame(gpsw = .$gpsw, gpsn = .$gpsn)), "\\w+$" ))) %>% 
  mutate(county = as.factor(str_to_title(county))) %>% 
  select(swab_label, county) -> county.df 

#add county data to metadata 
meta.county <- all.meta %>% 
  mutate(taxon = paste(genus, species, sep = " "),
         ) %>% 
  relocate(swab_label, taxon, date, site, oo_present, clinical_signs, log_copy_number, gpsn, gpsw, notes, collector)  %>% 
  left_join(county.df, by = "swab_label") %>% 
  mutate(county = ifelse(is.na(county.x) | county.x == "", as.character(county.y),
                         ifelse(as.character(county.x) != as.character(county.y) & !is.na(county.y), as.character(county.y), as.character(county.x))),
         .after = year) %>% 
  select(-c(county.x, county.y))

#annotate each taxon with ecomode 
meta.county %>% 
  select(taxon, ecomode) %>% 
  group_by(taxon, ecomode) %>% 
  summarize(n = n()) %>% 
  arrange(taxon, ecomode, n) %>% 
  group_by(taxon) %>%
  mutate(ecomode = fct_rev(ecomode)) %>% 
  arrange(taxon, ecomode) %>% 
  slice_head(n = 1) %>% 
  ungroup() %>% 
  mutate(ecomode = ifelse(taxon == "Cemophora coccinea", "Terrestrial Fossorial",
                          ifelse(taxon == "Farancia abacura", "Aquatic", as.character(ecomode)))) %>% 
  select(-n) %>% 
  mutate(across(.fns = ~as.factor(.x))) %>% 
  left_join(select(meta.county, -ecomode), .) %>% 
  mutate(swab_label = fct_recode(swab_label, `CMFP39.1` = "CMFP39")) %>% 
  relocate(ecomode, .after = taxon) %>% 
  relocate(notes, .after = last_col()) %>% 
  relocate(date, county, site, .after = oo_present) %>% 
  arrange(date) -> meta.final

#remove empty rows & columns from otu data before final join
otu.mat.rare %>% 
  as.data.frame() %>% 
  rownames_to_column("swab_label") %>% 
  filter(swab_label %in% meta.final$swab_label) %>%
  column_to_rownames("swab_label") %>% 
  select_if(colSums(.) > 0) %>% 
  filter(rowSums(.) !=0) -> otu.mat.final 

#filter taxonomy data final time 
tax.final <- tax.final %>% 
  filter(otu %in% colnames(otu.mat.final))

#join metadata with otu data 
otu.mat.final %>% 
  as.data.frame() %>% 
  rownames_to_column("swab_label") %>%
  right_join(meta.final, ., by = "swab_label") -> final.df

#export data as .csv objects  
write.csv(tax.final, file = paste(date, "SFDtaxfinal.csv", sep = "_"), row.names = F) #taxonomy data 
write.csv(final.df, file = paste(date, "SFDdatafull.formatted_ASR.csv", sep = "_"), row.names = F) #metadata and OTU data 
