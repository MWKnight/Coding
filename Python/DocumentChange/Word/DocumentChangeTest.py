import docx
doc = docx.Document()

doc.add_paragraph('Hello World!')
doc.save('helloWorld.docx')


