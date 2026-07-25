#comp = norm*(a*arr1 + b*arr2+ c*arr3 + d*arr4)
#0 <= a, b, c, d <= 1  ugy hogy a+b+c+d=1
class compositionHandler():
    def model(self, x, comp0, comp1, comp2, comp3):
        comp = comp0 * 1 + comp1 * 0 + comp2 * 0 + comp3 * 0
        return comp
    