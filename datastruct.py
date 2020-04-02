import operator


class Heap(object):
    """
    Yet another implementation of heaps.
    """

    def __init__(self, L, minheap=True):
        """
        Constructor that calls self.heapify().
        Args:
        L - list of values to heapify.
        minheap - True means a minimum heap and False means a maximum heap.
        """
        self.values = L
        # set up comparison operators for bubbling
        if minheap:
            self.operators = {
                "up": operator.lt,
                "down": operator.gt,
                "cross": operator.le,
            }
        else:
            self.operators = {
                "up": operator.gt,
                "down": operator.lt,
                "cross": operator.ge,
            }
        self.heapify()

    def heapify(self):
        """
        The heapify operation, which modifies self.values in-place at O(n).
        """
        n = len(self.values)
        for i in range(1, n):
            self.__bubbleup(i)

    def extract(self, lookup=False):
        """
        The extract operation, which returns the minimum or the maximum of the heap, depending on which type the heap is.
        The extracted value is then removed by default; setting lookup=True will override that and keep the value.
        """
        if len(self.values) < 1:
            return None
        retval = self.values[0]
        if not lookup:
            self.delete(0)
        return retval

    def insert(self, val):
        """
        The insert operation, which adds an element at the end and then bubbles it up.
        Args:
        val - the value to be inserted.
        """
        n = len(self.values)
        self.values.append(val)
        self.__bubbleup(n)

    def delete(self, i):
        """
        The delete operation, which replaces an element-by-index by the last element, removes the last element, and then bubbles the replaced element down.
        """
        n = len(self.values)
        # sanity check that the supplied index is valid
        assert i < n
        # replace the value to be removed
        self.values[i] = self.values[-1]
        # remove the last element
        self.values.pop(-1)
        self.__bubbledown(i)

    def __bubbleup(self, i):
        """
        The bubble-up operation to keep the heap invariant.
        Args:
        i - the index of the element to be bubbled.
        """
        # base case: if selected the root for bubble-up, just return
        if i < 1:
            return
        # find the index of the parent node of node i
        j = (i - 1) // 2
        # swap node i and its parent if the value of the node is less than that of its parent
        if self.operators["up"](self.values[i], self.values[j]):
            self.values[i], self.values[j] = self.values[j], self.values[i]
            # recursive call to bubble further
            self.__bubbleup(j)
        else:
            return

    def __bubbledown(self, i):
        """
        The bubble-down operation to keep the heap invariant.
        Args:
        i - the index of the element to be bubbled.
        """
        # find the indices of the child nodes of node i
        l = i * 2 + 1
        r = l + 1
        # base case: if selected a leaf node for bubble-down, just return
        n = len(self.values)
        if l >= n:
            return
        # special case: if only left child is available, check left child
        elif r >= n:
            # swap node i and its left child if the value of the node is greater than that of its left child
            if self.operators["down"](self.values[i], self.values[l]):
                self.values[i], self.values[l] = self.values[l], self.values[i]
                # recursive call to bubble further
                self.__bubbledown(l)
            else:
                return
        # common case: if both children are availble, check both and swap with the one with the smaller value
        else:
            # check whether node i should be swapped with its left child
            if self.operators["down"](
                self.values[i], self.values[l]
            ) and self.operators["cross"](self.values[l], self.values[r]):
                self.values[i], self.values[l] = self.values[l], self.values[i]
                # recursive call to bubble further
                self.__bubbledown(l)
            # check whether node i should be swapped with its right child
            elif self.operators["down"](
                self.values[i], self.values[r]
            ) and self.operators["cross"](self.values[r], self.values[l]):
                self.values[i], self.values[r] = self.values[r], self.values[i]
                # recursive call to bubble further
                self.__bubbledown(r)
            else:
                return
