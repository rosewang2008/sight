# SIGHT: A Large Annotated Dataset on Student Insights Gathered from Higher Education Transcripts
Authors: Rose E. Wang*, Pawan Wirawarn*, Noah Goodman and Dorottya Demszky

 *= Equal contributions

In the Proceedings of Innovative Use of NLP for Building Educational Applications 2023

Code & data coming soon!! 

## Motivation
Lectures are a learning experience for both students and teachers.
Students learn from teachers about the subject material, while teachers learn from students about how to refine their instruction.
Unfortunately, online student feedback is unstructured and abundant, making it challenging for teachers to learn and improve. We take a step towards tackling this challenge.
First, we contribute a dataset for studying this problem: SIGHT is a large dataset of 288 math lecture transcripts and 15,784 comments collected from the Massachusetts Institute of Technology OpenCourseWare (MIT OCW) YouTube channel.
Second, we develop a rubric for categorizing feedback types using qualitative analysis. 
Qualitative analysis methods are powerful in uncovering domain-specific insights, however they are costly to apply to large data sources.
To overcome this challenge, we propose a set of best practices for using large language models (LLMs) to cheaply classify the comments at scale.
We observe a striking correlation between the model's and humans' annotation: 
Categories with consistent human annotations (> $0.9$ inter-rater reliability, IRR) also display higher human-model agreement (> $0.7$ ), while categories with less consistent human annotations ( $0.7$ - $0.8$ IRR) correspondingly demonstrate lower human-model agreement ( $0.3$ - $0.5$ ).
These techniques uncover useful student feedback from thousands of comments, costing around \$0.002 per comment.
We conclude by discussing exciting future directions on using online student feedback and improving automated annotation techniques for qualitative research.
