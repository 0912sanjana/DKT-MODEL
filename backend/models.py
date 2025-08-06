from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, create_engine
from sqlalchemy.orm import declarative_base, relationship, scoped_session, sessionmaker
from datetime import datetime

# ✅ Base & DB Setup
Base = declarative_base()
engine = create_engine('sqlite:///database.db', connect_args={"check_same_thread": False})
Session = scoped_session(sessionmaker(bind=engine))
db_session = Session()

# ✅ Course Table
class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    questions = relationship("Question", back_populates="course")

# ✅ Student Table
class Student(Base):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

# ✅ Question Table
class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=True)  # 👈 used for answer comparison and display
    topic = Column(String, nullable=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    bloom_level = Column(String)

    course = relationship("Course", back_populates="questions")

# ✅ StudentResult Table with DKT, Bloom and Time
class StudentResult(Base):
    __tablename__ = 'student_results'

    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('students.id'))
    question_id = Column(Integer, ForeignKey('questions.id'))
    is_correct = Column(String, nullable=False)
    bloom_level = Column(String)
    dkt_score = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student")
    question = relationship("Question")

# ✅ Init Function
def init_db():
    Base.metadata.create_all(engine)
