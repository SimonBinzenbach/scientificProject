# KI-Abgabe
## Abgabe Künstliche Intelligenz von Simon Binzenbach und Oliver Pruchnicki

Link zum Repository: https://github.com/SimonBinzenbach/scientificProject

Das Projekt beinhaltet eine Python-Datei namens `main`, in der ein Convolutional Neural Network mit TensorFlow, jedoch ohne Keras, geschrieben wurde. 
Ziel des Projekts war es, Bilder von Katzen und Hunden zu unterscheiden. Das CNN funktioniert und kann exportiert bzw. geladen werden. Allerdings enthält das verwendete Dataset korrupte Bilder,
welche das Netzwerk zwar robuster machen, jedoch den Prozess des Nachvollziehens erschweren.

Das CNN zeigt Anzeichen von Overfitting (positive, explosive Validation Accuracy), was zum einen an den korrupten Bildern liegen kann, aber ebenso daran, dass wir zeitlich keinen Dropout-Algorithmus integrieren konnten.

Das Projekt ist debug- und ausführbar:

Ab Zeile 324 abwärts lässt sich die Struktur des Netzes anpassen. Wichtig ist hierbei, genau einen Convolutional Layer zu haben und die "output_size" exakt den möglichen Klassifizierungen entspricht (im Fall des aktuellen Datasets: 2).
Größe, Anzahl und Aktivierungsfunktion der Dense-Layer sind beliebig anpassbar. Für unseren Zweck haben wir 3 Dense-Layer, die an Komplexität abnehmen und jeweils die ReLU-Aktivierungsfunktion verwenden, da diese sich für die Bilderkennung gut eignet.
Der letzte Dense-Layer mit binärem Output nutzt die Sigmoid-Aktivierungsfunktion, da diese sich für binäre Outputs besonders eignet.

Ab Zeile 335 abwärts lassen sich die relevanten Funktionen für das Training anpassen:

	Loss Functions: cross_entropy_loss() -> funktioniert
			binary_cross_entropy_loss() -> funktioniert nicht

	Acc Function: 	accuracy() -> funktioniert
	
	Optimizer: 	Adam() -> funktioniert
			GradientDescent() -> funktioniert

Die einzige anpassbare Option hier ist der Optimizer, wobei wir standardmäßig Adam genommen haben, da dieser zwar langsamer, aber recht zuverlässig arbeitet.

Pipeline:
Input-Batch aus RGB-Bildern

Sobel-Edge-Detection: Da das verwendete Dataset variierende Bildgrößen hat, resizen wir das Bild und führen anschließend für jeden Farbkanal einen Sobel-Algorithmus aus. Die Ergebnisse werden zusammengeführt,
und wir erhalten einen Tensor mit 2 statt 3 Dimensionen (den Batch mal außen vor gelassen).

Max-Pooling: Auf den neuen Tensor wird anschließend ein Max-Pooling-Algorithmus angewendet, um den Input für das Dense-Layer weiter zu verringern.

Preprocessing: Bevor der Tensor an das Dense-Layer gegeben wird, normalisieren wir ihn und reduzieren ihn erneut um eine Dimension.

Dense-Layer Nr. 1

...

Dense-Layer Nr. n

Output
