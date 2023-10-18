CLEAR_STOCKS_FILES = False
stocks_folder = "data/stocks"
start = "2010-01-04"
end = "2022-12-30"



healthcare_companies = [
    "ABEO", "ABMD", "ABT", "ABUS", "ACAD", "ACHC", "ACHV", "ACIU", "ACOR", "ACRS",
    # "ACRX", "ADAP", "ADIL", "ADMA", "ADMP", "ADMS", "ADPT", "ADRO", "ADUS", "ADVM",
    # "AEMD", "AERI", "AEZS", "AFMD", "AGEN", "AGIO", "AGLE", "AGRX", "AGTC", "AHPI",
    # "AIMT", "AKAO", "AKBA", "AKCA", "AKER", "AKRO", "AKRX", "AKTX", "ALBO", "ALC",
    # "ALDR", "ALDX", "ALEC", "ALGN", "ALIM", "ALKS", "ALLK", "ALLO", "ALNA", "ALNY",
    # "ALPN", "ALRN", "ALSK", "ALT", "ALTM", "ALXN", "ALXO", "AMAG", "AMGN", "AMPE",
    # "AMPH", "AMRN", "AMRX", "AMYT", "ANAB", "ANIK", "ANIP", "ANIX", "ANPC", "ANTE",
    # "ANTM", "APLS", "APOG", "APOP", "APPN", "APTO", "APVO", "APYX", "AQST", "ARAV",
    # "ARCT", "ARDS", "ARDX", "AREC", "ARGX", "ARNA", "ARPO", "ARQT", "ARVN", "ARWR",
    # "ASLN", "ASMB", "ASND", "ASRT", "ASRV", "ASTC", "ATEC", "ATEX", "ATHE", "ATHX",
    # "ATNX", "ATOS", "ATRA", "ATRC", "ATRI", "ATRS", "ATXI", "AUPH", "AUTL", "AVDL",
    # "AVGR", "AVIR", "AVNS", "AVRO", "AVXL", "AXDX", "AXGN", "AXGT", "AXNX", "AXSM",
    # "AYTU", "AZN", "AZRX", "BABY", "BASI", "BAX", "BBH", "BBI", "BBIO", "BCLI",
    # "BCRX", "BDSI", "BDTX", "BEAM", "BHC", "BHVN", "BIB", "BIMI", "BIO", "BIOC",
    # "BIOL", "BIOS", "BIVI", "BJRI", "BLCM", "BLFS", "BLI", "BLPH", "BLRX", "BLUE",
    # "BMRA", "BMRN", "BMY", "BNGO", "BNTX", "BPTH", "BPMC", "BTAI", "BVXV", "BWAY",
    # "BYSI", "CABA", "CALA", "CANF", "CAPR", "CARA", "CARE", "CASI", "CBAY", "CBIO",
    # "CBMG", "CBPO", "CBTX", "CCXI", "CDNA", "CDTX", "CDXC", "CEMI", "CERC", "CERS",
    # "CETX", "CFMS", "CGC", "CGEN", "CGIX", "CHEK", "CHFS", "CHMA", "CHRS", "CHRW",
    # "CI", "CIDM", "CJJD", "CKPT", "CLBS", "CLDX", "CLGN", "CLLS", "CLRB", "CLSD",
    # "CLSN", "CLVS", "CLXT", "CMRX", "CMTL", "CNCE", "CNCM", "CNMD", "CNST", "CNTG",
    # "CNXN", "COCP", "CODX", "COHR", "COLL", "CORV", "CORT", "COST", "COUP", "CPHI",
    # "CPIX", "CPRX", "CRBP", "CRDF", "CREE", "CRIS", "CRMD", "CRNX", "CRSP", "CRSR",
    # "CRTX", "CRVS", "CRWD", "CSBR", "CSCO", "CSII", "CSL", "CSLLY", "CSPR", "CSTL",
    # "CTIC", "CTMX", "CTRE", "CTRV", "CTSO", "CTXR", "CUE", "CUTR", "CVET", "CVGI",
    # "CVS", "CVU", "CWBR", "CWCO", "CYAD", "CYAN", "CYCC", "CYCN", "CYH", "CYRX",
    # "CYTK", "DARE", "DBVT", "DCPH", "DDOG", "DERM", "DFFN", "DFPH", "DGX", "DHR",
    # "DMTK", "DNLI", "DOCU", "DRNA", "DRRX", "DSCI", "DVA", "DVAX", "DXCM", "DYNT",
    # "EARS", "EAST", "EBAY", "EBS", "EBSB", "ECHO", "ECOR", "EDAP", "EDIT", "EDNT",
    # "EDSA", "EIDX", "EIGR", "EKSO", "ELAN", "ELOX", "ELSE", "ELVT", "EMAN", "ENDP",
    # "ENLV", "ENOB", "ENSG", "ENTA", "ENTG", "ENTX", "ENZ", "EOLS", "EPZM", "EQ",
    # "ERYP", "ESPR", "ETON", "ETTX", "EUCR", "EVBG", "EVFM", "EVGN", "EVH", "EVLO",
    # "EVOK", "EVOP", "EVTI", "EW", "EYEG", "EYEN", "EYPT", "FAMI", "FATE", "FBIO",
    # "FBIOP", "FBMS", "FCNCA", "FENC", "FGEN", "FHB", "FHTX", "FIBK", "FISI", "FISV",
    # "FITB", "FITBI", "FIVE", "FIVN", "FLDM", "FLGT", "FLIC", "FMAO", "FMBH", "FMBI",
    # "FMBIO", "FMBIP", "FMNB", "FOLD", "FORM", "FORR", "FOSL", "FOX", "FOXA", "FOXF",
    # "FPRX", "FRBK", "FREQ", "FRG", "FRGI", "FRME", "FROG", "FRPH", "FRSX", "FRTA",
    # "FSBW", "FSLR", "FSRV", "FSRVU", "FSRVW", "FST", "FSTR", "FTEK", "FTFT", "FTHM",
    # "FTNT", "FTSV", "FULC", "FULT", "FUNC", "FUSN", "FUTU", "FUV", "FVAM", "FVCB",
    # "FVE", "FWONA", "FWONK", "FXNC", "GABC", "GAIA", "GAIN", "GAINL", "GAINM", "GALT",
    # "GARS", "GASS", "GBCI", "GBDC", "GBIO", "GBLI", "GBLIL", "GBLIZ", "GBT", "GCBC",
    # "GCMG", "GCMGW", "GDEN", "GDRX", "GDS", "GDYN", "GDYNW", "GECC", "GECCL", "GECCM",
    # "GECCN", "GENC", "GENE", "GENY", "GEOS", "GERN", "GEVO", "GFED", "GFN", "GFNCP",
    # "GFNSL", "GGAL", "GH", "GHIV", "GHIVU", "GHIVW", "GHSI", "GIFI", "GIGM", "GIII",
    # "GILD", "GILT", "GLAD", "GLADD", "GLADL", "GLBS", "GLBZ", "GLDD", "GLDI", "GLG",
    # "GLMD", "GLNG", "GLOP", "GLOPP", "GLPG", "GLPI", "GLRE", "GLSI", "GLTO", "GLUU",
    # "GLYC", "GMAB", "GMDA", "GMHI", "GMHIU", "GMHIW", "GMLP", "GMLPP", "GNCA", "GNFT",
    # "GNMK", "GNPX", "GNRS", "GNRSU", "GNRSW", "GNSS", "GNTX", "GNTY", "GNUS", "GO",
    # "GOCO", "GOGL", "GOGO", "GOOD", "GOODM", "GOODN", "GOOG", "GOOGL", "GOSS", "GPOR",
    # "GPP", "GPRE", "GPRO", "GRBK", "GRFS", "GRID", "GRIF", "GRIL", "GRIN", "GRMN",
    # "GRNQ", "GROW", "GRPN", "GRSV", "GRSVU", "GRSVW", "GRTS", "GRTX", "GRVY", "GRWG",
    # "GSBC", "GSHD", "GSIT", "GSK", "GSKY", "GSM", "GSMG", "GSMGW", "GSUM", "GT",
    # "GTEC", "GTH", "GTHX", "GTIM", "GTLS", "GTYH", "GURE", "GVP", "GWGH", "GWPH",
    # "GWRS", "GXGX", "GXGXU", "GXGXW", "GYRO", "HA", "HAFC", "HAIN", "HALL", "HALO",
    # "HAPP", "HARP", "HAS", "HAYN", "HBAN", "HBANN", "HBANO", "HBCP", "HBIO", "HBMD",
    # "HBNC", "HBP", "HBT", "HCAC", "HCACU", "HCACW", "HCAP", "HCAPZ", "HCAT", "HCCO",
    # "HCCOU", "HCCOW", "HCCI", "HCKT", "HCM", "HCSG", "HDS", "HDSN", "HEAR", "HEBT",
    # "HEC", "HECCU", "HECCW", "HEES", "HELE", "HEPA", "HEPZ", "HERD", "HFBL", "HFFG",
    # "HFWA", "HGSH", "HHR", "HIBB", "HIFS", "HIHO", "HIIQ", "HIMX", "HJLI", "HJLIW",
    # "HLG", "HLIO", "HLIT", "HLNE", "HMHC", "HMNF", "HMST", "HMSY", "HMTV", "HNDL",
    # "HNNA", "HNRG", "HOFT", "HOL", "HOLI", "HOLUU", "HOLUW", "HOLX", "HOMB", "HONE",
    # "HOOK", "HOPE", "HOTH", "HOVNP", "HQI", "HQY", "HRMY", "HROW", "HRTX", "HRZN",
    # "HSDT", "HSIC", "HSII", "HSKA", "HSON", "HSTM", "HTBI", "HTBK", "HTBX", "HTGM",
    # "HTHT", "HTLD", "HTLF", "HUBG", "HUGE", "HUIZ", "HURC", "HURN", "HVBC", "HWBK",
    # "HWC", "HWCC", "HWCPL", "HWKN", "HX", "HYAC", "HYACU", "HYACW", "HYLS", "HYMC",
    # "HYMCL", "HYMCW", "HYMCZ", "HYRE", "IAC", "IART", "IBB", "IBCP", "IBEX", "IBKR",
    # "IBM", "IBOC", "IBTA", "IBTB", "IBTD", "IBTE", "IBTF", "IBTG", "IBTH", "IBTI",
    # "IBTJ", "IBTK", "IBTX", "IBUY", "ICAD", "ICBK", "ICCC", "ICCH", "ICFI", "ICHR",
    # "ICLK", "ICLN", "ICLR", "ICMB", "ICON", "ICPT", "ICUI", "IDCC", "IDEX", "IDN",
    # "IDRA", "IDXG", "IDXX", "IDYA", "IEA", "IEAWW", "IEC", "IEF", "IEI", "IAC", "IART",
    # "IBB", "IBCP", "IBEX", "IBKR", "IBM", "IBOC", "IBTA", "IBTB", "IBTD", "IBTE",
    # "IBTF", "IBTG", "IBTH", "IBTI", "IBTJ", "IBTK", "IBTX", "IBUY", "ICAD", "ICBK",
    # "ICCC", "ICCH", "ICFI", "ICHR", "ICLK", "ICLN", "ICLR", "ICMB", "ICON", "ICPT",
    # "ICUI", "IDCC", "IDEX", "IDN", "IDRA", "IDXG", "IDXX", "IDYA", "IEA", "IEAWW",
    # "IEC", "IEF", "IEI", "IEMG", "IEP", "IESC", "IEUS", "IFEU", "IFGL", "IFMK",
    # "IFRX", "IFV", "IGF", "IGIB", "IGOV", "IGSB", "IHRT", "III", "IIIN", "IIIV",
    # "IIN", "IIVI", "IKNX", "ILMN", "ILPT", "IMAB", "IMAC", "IMACW", "IMBI", "IMGN",
    # "IMKTA", "IMMP", "IMMR", "IMMU", "IMOS", "IMRA", "IMRN", "IMRNW", "IMTE", "IMTX",
    # "IMTXW", "IMUX", "IMV", "IMVT", "IMXI", "INAQ", "INAQU", "INAQW", "INBK", "INBKL",
    # "INBKZ", "INCY", "INDB", "INDY", "INFI", "INFN", "INFO", "INFR", "INGN", "INMB",
    # "INMD", "INNT", "INO", "INOD", "INOV", "INPX", "INSE", "INSG", "INSM", "INTC",
    # "INTG", "INTL", "INTU", "INVA", "INVE", "INVO", "INZY", "IONS", "IOSP", "IOTS",
    # "IOVA", "IPAR", "IPDN", "IPGP", "IPHA", "IPKW", "IPLDP", "IPWR", "IQ", "IRBT",
    # "IRCP", "IRDM", "IRIX", "IRMD", "IROQ", "IRTC", "IRWD", "ISBC", "ISDS", "ISDX",
    # "ISEE", "ISEM", "ISHG", "ISIG", "ISNS", "ISRG", "ISSC", "ISTB", "ISTR", "ITCI",
    # "ITI", "ITIC", "ITMR", "ITOS", "ITRI", "ITRM", "ITRN", "IVA", "IVAC", "IVC",
    # "IVENC", "IVFGC", "IVFVC", "IVR", "IXUS", "IZEA", "JACK", "JAGX", "JAKK", "JAMF",
    # "JAN", "JAZZ", "JBHT", "JBLU", "JBSS", "JCOM", "JCS", "JCTCF", "JD", "JFIN",
    # "JFU", "JG", "JIH", "JIHUU", "JIHUW", "JJSF", "JKHY", "JKI", "JMPNL", "JMPNZ",
    # "JNCE", "JOBS", "JOUT", "JRJC", "JRSH", "JRVR", "JSM", "JSMD", "JSML", "JVA",
    # "JYNT", "KALA", "KALU", "KALV", "KBAL", "KBLM", "KBLMR", "KBLMU", "KBLMW", "KBSF",
    # "KBWB", "KBWD", "KBWP", "KBWR", "KBWY", "KCAPL", "KDMN", "KDNY", "KDP", "KE",
    # "KELYA", "KELYB", "KEQU", "KERN", "KERNW", "KFFB", "KFRC", "KHC", "KIDS", "KIN",
    # "KINS", "KINZ", "KINZU", "KINZW", "KIRK", "KLAC", "KLDO", "KLIC", "KLXE", "KMDA",
    # "KMPH", "KNDI", "KNSA", "KNSL", "KOD", "KOPN", "KOR", "KOSS", "KPTI", "KRBP",
    # "KRKR", "KRMD"
    ]

tech_companies = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "PYPL", "INTC", "CSCO",
    # "ADBE", "NFLX", "ORCL", "IBM", "QCOM", "SAP", "TXN", "AVGO", "BIDU", "JD",
    # "UBER", "CTSH", "INTU", "AMD", "VMW", "HPQ", "NOW", "TWTR", "TEAM", "SNAP",
    # "EA", "ADSK", "CRWD", "ZM", "DOCU", "OKTA", "DDOG", "ATVI", "SPLK", "MRVL",
    # "ANET", "FTNT", "VRSN", "AKAM", "ZS", "LYFT", "WDC", "CDNS", "CDW", "STX",
    # "KLAC", "SWKS", "GLW", "GRMN", "TEL", "APH", "IPGP", "KEYS", "MXIM", "COHR",
    # "SNPS", "TER", "ANSS", "VRSK", "LRCX", "BR", "FISV", "NLOK", "JKHY", "LDOS",
    # "IT", "GPN", "FLT", "FIS", "EPAM", "DXC", "CTAS", "CERN", "BKI", "ANET",
    # "ADI", "ACN", "PAYC", "TYL", "RNG", "PTC", "FICO", "FFIV", "SSNC", "MIME",
    # "JCOM", "GWRE", "DBX", "CTXS", "BL", "AYX", "AVLR", "APPN", "ALRM", "ZEN",
    # "XM", "WORK", "WIX", "TWLO", "TTD", "SMAR", "SHOP", "SGEN", "PVTL", "PLT",
    # "PANW", "NTNX", "NET", "MDB", "LOGM", "KXS", "HUBS", "GDDY", "FEYE", "ESTC",
    # "ENSG", "EGHT", "DOYU", "CRWD", "BOX", "AYI", "APPS", "AL", "AKAM", "ADSK",
    # "ADP", "YEXT", "XRX", "WP", "UPLD", "TXN", "TSE", "SQ", "SPSC", "SNX",
    # "SIVB", "SIRI", "SINA", "SHI", "SGH", "RHT", "RAMP", "QUOT", "PSTG", "PRGS",
    # "PCTY", "OTEX", "ON", "OMCL", "NTAP", "NCR", "MTCH", "MPWR", "MCHP", "LITE",
    # "LECO", "LDOS", "LBRDK", "LAMR", "KODK", "KLIC", "JKHY", "JBL", "ITRI", "IPHI",
    # "INSG", "IMMR", "HPE", "HIMX", "GRUB", "GPRO", "GNMK", "FLIR", "FLEX", "FEYE",
    # "FARO", "EXTR", "ENPH", "DIOD", "CVLT", "CREE", "CCMP", "CACI", "BRKS", "BPOP",
    # "BHE", "AMKR", "ALTR", "ALLT", "AOSL", "AMBA", "ALRM", "AIRG", "AEIS", "ACMR"
]

finance_companies = [
    "CME", "SPGI", "MSCI", "MCO", "SIVB", "FITB", "ETFC", "MKTX", "RF", "CBOE",
    # "HBAN", "NDAQ", "ZION", "FRC", "PBCT", "PACW", "TFSL", "CINF", "ARES", "IBKR",
    # "SEIC", "FULT", "WAL", "GBCI", "TCBI", "SSB", "FCFS", "BPFH", "CM", "PFG",
    # "RJF", "AON", "TROW", "AMP", "AJG", "IVZ", "AIZ", "NTRS", "FHN", "JHG",
    # "CIT", "CFG", "KEY", "BEN", "HBNC", "STT", "MTB", "USB", "ALL", "TRV",
    # "PGR", "AFL", "WFC", "GS", "MS", "DFS", "SYF", "COF", "BAC", "JPM",
    # "C", "BRK-B", "MET", "PRU", "LNC", "UNM", "WRB", "CNO", "RE", "L",
    # "ORI", "MCY", "PRA", "THG", "KINS", "TIPT", "RLI", "SIGI", "NGHC", "OB",
    # "HCI", "HALL", "FNHC", "EIG", "DGICA", "CINF", "CB", "AXS", "ANAT", "AFG",
    # "ACGL", "WTM", "RNR", "PTP", "PRA", "PRE", "NAV", "MHLD", "MBI", "KMPR",
    # "HIG", "ENH", "EMCI", "CNA", "BWINB", "AGO", "AGII", "AEH", "WLTW", "UIHC",
    # "TRV", "TPRE", "TCHC", "STFC", "SAFT", "RLI", "RGA", "RE", "RDN", "PTLA",
    # "PRA", "PPBI", "PIH", "OXLC", "NWLI", "NMIH", "NAVG", "MRLN", "MLVF", "MKL",
    # "MCY", "MCBC", "MBWM", "MBIN", "LTXB", "LSBK", "LMRK", "LION", "LBAI", "KINS",
    # "ISBC", "IBN", "HRTG", "HTLF", "HTH", "HRZN", "HFWA", "HCC", "HAS", "GWB",
    # "GTS", "GRIF", "GNTY", "GLRE", "FULT", "FMBH", "FISI", "FHN", "FFIN", "FCNCA",
    # "FCBC", "FARO", "EWBC", "ENVA", "EBMT", "EA", "DGICA", "DFS", "DBD", "CSFL",
    # "CSBK", "COWN", "CPF", "COF", "CMCT", "CHCO", "CATY", "CASY", "CARV", "CACC",
    # "BUSE", "BXS", "BWFG", "BWINB", "BPFH", "BOKF", "BHLB", "BGCP", "BCBP", "BANR",
    # "BANF", "ASRV", "AROW", "AMNB", "ALRS"
]

manufacturing_companies = [
    "AAL", "ABMD", "AOSL", "ACMR", "ADI", "ADP", "ADSK", "AERI", "AGIO", "AIMC",
    # "AIRG", "ALCO", "ALGT", "ALXN", "AMAT", "AMBA", "AMKR", "AMOT", "AMPH", "AMSC",
    # "AMWD", "ANAB", "ANSS", "AOSL", "APEN", "AREX", "ARLP", "ARTNA", "ASTE", "ATEC",
    # "ATRA", "ATRO", "ATVI", "AVAV", "AXGN", "AXTI", "AZPN", "B", "BBSI", "BCPC",
    # "BDSI", "BEAT", "BGFV", "BIIB", "BLDP", "BLFS", "BOOM", "BREW", "BRID", "BRKS",
    # "BRSS", "BSET", "BUFF", "BWEN", "CAAS", "CAMP", "CASH", "CATM", "CENX", "CETX",
    # "CETXP", "CETXW", "CG", "CGEN", "CGNX", "CHTR", "CLAR", "CLNE", "CLVS", "CMCO",
    # "COHR", "CONE", "CORE", "CORT", "CREE", "CRUS", "CSGS", "CSII", "CSL", "CSWI",
    # "CTAS", "CTRL", "CVGI", "CWST", "CYBE", "CYRX", "CYRXW", "DAKT", "DCO", "DGLY",
    # "DIOD", "DJCO", "DLB", "DORM", "DPW", "DRTT", "DSPG", "DWSN", "DXCM", "EA",
    # "EFOI", "EGLT", "EHTH", "ELGX", "EMAN", "EMKR", "ENDP", "ENG", "ENPH", "ENTG",
    # "EVBG", "EXEL", "EXFO", "EXLS", "FARO", "FELE", "FET", "FEYE", "FFIV", "FLEX",
    # "FLIR", "FLOW", "FLXS", "FORK", "FORM", "FOSL", "FOX", "FOXF", "FPRX", "FRSX",
    # "FSTR", "FTEK", "FTK", "FULC", "FWRD", "GASS", "GBCI", "GBLI", "GCO", "GIII",
    # "GILD", "GKOS", "GLUU", "GNTX", "GOGL", "GRMN", "GSM", "GTLS", "GURE", "HLIO",
    # "HOLI", "HUBG", "ICHR", "IDCC", "IDRA", "IDXX", "IESC", "IFMK", "IGT", "IIN",
    # "IMGN", "IMI", "IMKTA", "IMMR", "IMXI", "INFN", "INGR", "INO", "INSE", "INTC",
    # "IOSP", "IPAR", "IPGP", "IRBT", "ISNS", "ISRG", "ISSC", "ITI", "ITRI", "JACK",
    # "JASN", "JASNW", "JAZZ", "JBSS", "JJSF", "JOUT", "KALU", "KELYA", "KELYB", "KEQU",
    # "KLIC", "KLXE", "KOPN", "KRNT", "LAC", "LAD", "LANC", "LAWS", "LAZY", "LBAI",
    # "LBRDA", "LBRDK", "LECO", "LEDS", "LFUS", "LGIH", "LITE", "LMB", "LPTH", "LSCC",
    # "LTRPA", "LTRPB", "LTRX", "LULU", "LUNA", "MAC", "MANT", "MARK", "MAT", "MATW",
    # "MCF", "MCHP", "MDCO", "MDLZ", "MEI", "MGPI", "MLAB", "MLHR", "MLNK", "MMSI",
    # "MNRO", "MOBL", "MOCO", "MPAA", "MPB", "MRCY", "MRNS", "MRTN", "MSTR", "MTSC",
    # "MU", "MXIM", "MYGN", "NANO", "NATH", "NBEV", "NCMI", "NDSN", "NEOG", "NEON",
    # "NEWA", "NK", "NKSH", "NMIH", "NNBR", "NOVT", "NSIT", "NTAP", "NTCT", "NTRA",
    # "NTRI", "NUAN", "NVEC", "NVMI", "NXPI", "OBCI", "OBNK", "OESX", "OFIX", "OLED",
    # "OLLI", "OMCL", "ON", "ONCY", "ONTO", "OPK", "OPTN", "ORA", "ORBC", "OSIS",
    # "OSPN", "OSS", "OSTK", "PAA", "PATK", "PAYX", "PCH", "PCMI", "PCRX", "PCTI",
    # "PCTY", "PEGA", "PERY", "PETS", "PI", "PICO", "PKOH", "PLAB", "PLCE", "PLPC",
    # "PLSE", "PLUS", "PLXS", "POWI", "PPC", "PRAA", "PRAH", "PRCP", "PRFT", "PRGS",
    # "PRMW", "PROV", "PRPH", "PRTS", "PSEC", "PXLW", "QADA", "QADB", "QCOM", "QDEL",
    # "QTRX", "QUIK", "RBBN", "RBCAA", "RBCN", "RCII", "RCKY", "RDI", "RDIB", "RECN",
    # "RESN", "RETA", "REXR", "RFIL", "RGEN", "RGLD", "RICK", "RIGL", "RMBL", "ROCK",
    # "ROG", "ROST", "RP", "RPD", "RTRX", "RUN", "RUSHA", "RUSHB", "RVSB", "RXN",
    # "SABR", "SAFM", "SANM", "SANW", "SATS", "SAUC", "SBBP", "SBCF", "SBGI", "SBLK",
    # "SBNY", "SBRA", "SBSI", "SBUX", "SCHN", "SCKT", "SCON", "SCPL", "SCSC", "SCVL",
    # "SCWX", "SDC", "SEAC", "SEIC", "SENEA", "SENEB", "SFBS", "SFNC", "SGA", "SGEN",
    # "SGH", "SGMO", "SGMS", "SHEN", "SHLO", "SHOO", "SHOS", "SHPG", "SHSP", "SIGA",
    # "SILC", "SINA", "SIRI", "SIVB", "SKOR", "SKYS", "SKYW", "SLAB", "SLGN", "SLM",
    # "SLP", "SMBC", "SMP", "SMPL", "SMRT", "SMSI", "SMTC", "SNBR", "SNCR", "SNDR",
    # "SNHY", "SNPS", "SOHU", "SONA", "SP", "SPAR", "SPCB", "SPKE", "SPLK", "SPNE",
    # "SPNS", "SPPI", "SPRO", "SPSC", "SPTN", "SPWH", "SQBG", "SRCE", "SRCL", "SRDX",
    # "SRE", "SRNE", "SRPT", "SSB", "SSBI", "SSNC", "SSNT", "SSP", "SSRM", "SSTI",
    # "SSYS", "STAA", "STBA", "STCN", "STFC", "STKL", "STLD", "STML", "STMP", "STRA",
    # "STRL", "STRS", "STRT", "STX", "SUMR", "SUNS", "SUPN", "SVBI", "SWKS", "SYBT",
    # "SYKE", "SYMC", "SYNA", "SYNH", "SYNL", "SYPR", "SYRS", "TA", "TACO", "TACT",
    # "TANH", "TAST", "TATT", "TAYD", "TBBK", "TBIO", "TBK", "TBNK", "TBPH", "TCBI",
    # "TCBK", "TCCO", "TCDA", "TCF", "TCFC", "TCMD", "TCON", "TCPC", "TCRD", "TCRR",
    # "TCX", "TDIV", "TEAM", "TECD", "TECH", "TECTP", "TEDU", "TELL", "TENB", "TENX",
    # "TER", "TERP", "TESS", "TGA", "TGEN", "TGLS", "TGTX", "THFF", "THRM", "TILE",
    # "TIPT", "TISA", "TITN", "TIVO", "TLC", "TLF", "TLGT", "TLND", "TLRY", "TLSA",
    # "TLYS", "TMUS", "TNAV", "TNDM", "TNXP", "TOPS", "TORC", "TOWN", "TPCO", "TPIC",
    # "TPTX", "TRCH", "TREE", "TRHC", "TRIB", "TRIL", "TRIP", "TRMB", "TRMD", "TRMK",
    # "TRNS", "TROV", "TROW", "TRS", "TRST", "TRUE", "TRUP", "TRVG", "TRVN", "TSBK",
    # "TSC", "TSCO", "TSEM", "TSG", "TSRI", "TTEC", "TTEK", "TTGT", "TTMI", "TTNP",
    # "TTOO", "TTPH", "TTWO", "TUES", "TWIN", "TWMC", "TWOU", "TXMD", "TXN", "TXRH",
    # "TYHT", "TYME", "TYPE", "TZOO", "UAE", "UBCP", "UBFO", "UBOH", "UBSI", "UCBI",
    # "UCFC", "UCTT", "UEIC", "UEPS", "UFCS", "UFPI", "UFPT", "UG", "UHAL", "UIHC",
    # "ULBI", "UMBF", "UMPQ", "UNAM", "UNB", "UNFI", "UNIT", "UNTY", "UONE", "UONEK",
    # "URBN", "URGN", "USAK", "USAP", "USAT", "USATP", "USCR", "USEG", "USLM", "USLV",
    # "UTHR", "UTMD", "UVSP", "UXIN", "VALU", "VBFC", "VBIV", "VBLT", "VBTX", "VC",
    # "VCEL", "VCRA", "VCYT", "VECO", "VERI", "VERU", "VETS", "VFF", "VG", "VGIT",
    # "VGLT", "VGSH", "VIA", "VIAB", "VIAV", "VICR", "VIGI", "VIOT", "VIRC", "VIRT",
    # "VIVE", "VIVO", "VKTX", "VLGEA", "VLO", "VLRX", "VLY", "VMBS", "VNDA", "VNET",
    # "VNOM", "VNQI", "VOD", "VONE", "VONG", "VONV", "VOXX", "VRA", "VRAY", "VREX",
    # "VRIG", "VRML", "VRNS", "VRNT", "VRSK", "VRSN", "VRTS", "VRTSP", "VRTU", "VRTX",
    # "VSAT", "VSEC", "VSTM", "VTGN", "VTHR", "VTIP", "VTVT", "VTWG", "VTWO", "VTWV",
    # "VUSE", "VVPR", "VVUS", "VWOB", "VXUS", "VYGR", "VYMI", "WABC", "WAFD", "WASH",
    # "WATT", "WB", "WBA", "WCFB", "WDC", "WDFC", "WEBK", "WEN", "WERN", "WETF",
    # "WEYS", "WHF", "WHFBZ", "WHLM", "WHLR", "WHLRD", "WHLRP", "WIFI", "WILC", "WINA",
    # "WING", "WINS", "WIRE", "WISA", "WIX", "WKHS", "WLDN", "WLFC", "WLTW", "WMGI",
    # "WNEB", "WOOD", "WRLD", "WRTC", "WSBC", "WSBF", "WSC", "WSFS", "WSM", "WSTG",
    # "WSTL", "WTBA", "WTFC", "WTFCM", "WTRE", "WTREP", "WTRH", "WVE", "WVFC", "WVVI",
    # "WVVIP", "WWD", "WWR", "WYNN", "XAIR", "XBIO", "XBIT", "XCUR", "XEL", "XELB",
    # "XENE", "XENT", "XERS", "XFOR", "XGN", "XLNX", "XLRN", "XNCR", "XNET", "XOG",
    # "XOMA", "XONE", "XP", "XPEL", "XPER", "XRAY", "XSPA", "XT", "YGYI", "YI",
    # "YIN", "YLCO", "YLDE", "YMAB", "YNDX", "YORW", "YRCW", "YTEN", "YTRA", "YVR",
    # "Z", "ZAGG", "ZBRA", "ZEAL", "ZEUS", "ZG", "ZGNX", "ZION", "ZIONW", "ZIOP",
    # "ZIXI", "ZKIN", "ZLAB", "ZN", "ZNGA", "ZS", "ZUMZ", "ZYNE", "ZYXI"
]
    