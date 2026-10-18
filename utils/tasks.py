"""
Copied from Script for evaluating Jina Embedding Models on the MTEB benchmark.
https://github.com/embeddings-benchmark/mteb/blob/main/scripts/run_mteb_english.py
"""

TASK_LIST_STS = [
    "STS17",
    "SICK-R",
    "STSBenchmark",
    "BIOSSES",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS22",
]

TASK_LIST_CLASSIFICATION = [
    "Banking77Classification",
    "EmotionClassification",
    "MassiveIntentClassification",
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "ImdbClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "BiorxivClusteringS2S",
    "MedrxivClusteringS2S",
    "TwentyNewsgroupsClustering",
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "MedrxivClusteringP2P",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "SciDocsRR",
    "StackOverflowDupQuestions",
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
]

TASK_LIST_RETRIEVAL = [
    "SciFact",
    "ArguAna",
    "NFCorpus",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "FiQA2018",
    "QuoraRetrieval",
    "SCIDOCS",
    "Touche2020",
    "TRECCOVID",
    "MSMARCO",
    "NQ",
    "HotpotQA",
    "DBPedia",
    "FEVER",
    "ClimateFEVER",
]

TASK_LIST_SUMMARIZATION = [
    "SummEval",
]

ALL_TASKS = [
    ("STS", TASK_LIST_STS),
    ("Classification", TASK_LIST_CLASSIFICATION), 
    ("Clustering", TASK_LIST_CLUSTERING), 
    ("PairClassification", TASK_LIST_PAIR_CLASSIFICATION), 
    ("Reranking", TASK_LIST_RERANKING), 
    ("Retrieval", TASK_LIST_RETRIEVAL),
    ("Summarization", TASK_LIST_SUMMARIZATION),
]
