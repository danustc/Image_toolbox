"""
Created by Dan on 10/12/2016. An implimentation of linked list in Python.
"""
# first, let's think about a simple version of linked list (a real python list!)


class Simple_list(object):
    """
    The elements of the simple list:
    z_marker: from which z-position did we begin
    list: contains all the cell markers (indices, column 0) and average signal intensities (column 1)
    """

    def __init__(self, z_marker = 0):
        self.z_marker = z_marker
        self.list = []

    def add(self, new_node):
        # new_node: just a two-element list
        self.list.append(new_node)

# Next, let's think about a classical implementation of linked list (like those in C/C++).

class Node:

    def __init__(self,data):
        self.data = data
        self.next = None
        self.prev = None


class Linked_list(object):

    def __init__(self, z_marker = 0):
        self.zs = z_marker
        self.ll = 0 # link list length
        self.head = None
        self.tail = None


    def add(self, data):
        """
        This is slightly
        """
        node = Node(data)
        if self.head == None:
            self.head = node
            self.tail = node  # both head and tail points to the next
        else:
            # link together
            node.prev = self.tail
            node.prev.next = node
            self.tail = node # updated

        self.ll +=1


    def search(self, ndp):
        """
        search from the head or tail
        """
        pn = self.head
        if (pn!=None):
            dc = 0
            while(dc < ndp and pn.next !=None):
                pn = pn.next
                dc+=1

            print("The real depth:", dc) # print out the real depth that we went through
            return pn, dc
