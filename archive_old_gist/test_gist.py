from gist_extractive import GistExtractiveSummarizer

print("🔹 Loading model... please wait")

g = GistExtractiveSummarizer("models/Gist")

sample = """The petitioner challenges the order passed by the High Court
regarding property ownership dispute. The respondent claims prior possession
and relies on Section 110 of the Evidence Act. The trial court dismissed the
petition citing lack of evidence, while the appellate court reversed that
finding. The Supreme Court held that mere possession without title cannot
grant ownership rights, upholding the appellate order."""

print("\n🔹 Extractive Summary:")
print(g.summarize(sample, top_n=5))
